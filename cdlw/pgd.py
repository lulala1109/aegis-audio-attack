"""Shared layer-adaptive sign-PGD loop.

All methods use the same initialization, projection, budget and step schedule.
Only the internal objective and the layer weights differ.  The internal
gradient is rescaled to a fixed fraction (``lambda_internal``) of the base
gradient norm before the two are combined, so the internal term stays
meaningful regardless of the loss magnitudes (gradient balancing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .determinism import make_generator, seed_everything
from .io import snr_db, tensor_to_list
from .model import Layout, QwenAudioAttackModel
from .objectives import internal_loss

MAX_GRAD_SCALE = 10.0


@dataclass
class AttackConfig:
    eps: float = 0.01
    alpha: float = 0.001
    steps: int = 100
    kappa: float = 5.0
    random_init: bool = True
    lambda_internal: float = 0.1
    reference_prob: float = 0.02
    check_every: int = 0
    max_new_tokens: int = 256
    seed: int = 11
    eval_every: int = 0
    early_stop: bool = False
    momentum: float = 0.0
    select_best: bool = False


@dataclass
class AttackResult:
    original_wav: torch.Tensor
    adversarial_wav: torch.Tensor
    perturbation: torch.Tensor
    history: list[dict]
    final_base_loss: float
    final_internal_loss: float
    original_response: str | None = None
    adversarial_response: str | None = None
    success: bool | None = None
    success_step: int | None = None
    evaluated_steps: list[int] = field(default_factory=list)
    eval_history: list[dict] = field(default_factory=list)
    generation_calls: int = 0
    judge_calls: int = 0
    final_judge_label: str | None = None
    selected_step: int | None = None


def project_delta(delta: torch.Tensor, waveform: torch.Tensor, eps: float) -> torch.Tensor:
    """Project to the L_inf ball and to the valid waveform range."""
    delta = delta.clamp(-eps, eps)
    return (waveform + delta).clamp(-1.0, 1.0) - waveform


def run_pgd(
    model: QwenAudioAttackModel,
    waveform: torch.Tensor,
    layout: Layout,
    scheduler,
    config: AttackConfig,
    verbose: bool = True,
    evaluate_fn=None,
) -> AttackResult:
    seed_everything(config.seed)
    waveform = waveform.to(model.device, dtype=torch.float32).clamp(-1.0, 1.0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if config.random_init:
        generator = make_generator(waveform.device, config.seed)
        delta = torch.empty_like(waveform).uniform_(
            -config.eps, config.eps, generator=generator)
    else:
        delta = torch.zeros_like(waveform)
    delta = project_delta(delta, waveform, config.eps)

    use_internal = scheduler.mode != "none" and float(config.lambda_internal) > 0.0
    active_layers = list(scheduler.layers) if use_internal else []
    reference_logprob = model.reference_logprob(config.reference_prob)
    history: list[dict] = []
    eval_history: list[dict] = []
    evaluated_steps: list[int] = []
    success: bool | None = None
    success_step: int | None = None
    adversarial_response: str | None = None
    final_judge_label: str | None = None
    generation_calls = 0
    judge_calls = 0
    velocity: torch.Tensor | None = None
    best_delta: torch.Tensor | None = None
    best_base_loss: float | None = None
    selected_step: int | None = None

    for step in range(config.steps):
        delta = delta.detach().clone().requires_grad_(True)
        adversarial = waveform + delta
        pack = model.forward_attack(adversarial, layout, active_layers)

        refusal_logprobs = {
            layer: logprob.detach()
            for layer, logprob in pack.refusal_logprobs.items()
        }
        refusal_probs = {
            layer: logprob.exp()
            for layer, logprob in refusal_logprobs.items()
        }
        if use_internal:
            weights = scheduler.update(step, refusal_logprobs)
        else:
            weights = torch.zeros(0, device=model.device)

        if not use_internal:
            internal = pack.base_loss.new_zeros(())
        else:
            terms = []
            for index, layer in enumerate(scheduler.layers):
                logprob = pack.refusal_logprobs[layer]
                terms.append(weights[index] * internal_loss(logprob, reference_logprob))
            internal = torch.stack(terms).sum()

        # Gradient-balanced sign-PGD: keep the internal gradient at a fixed
        # fraction of the base-gradient norm instead of a fixed loss weight.
        if use_internal:
            grad_base = torch.autograd.grad(
                pack.base_loss, delta, retain_graph=True)[0]
            grad_internal = torch.autograd.grad(
                internal, delta, retain_graph=False)[0]
        else:
            grad_base = torch.autograd.grad(pack.base_loss, delta, retain_graph=False)[0]
            grad_internal = None
        if not torch.isfinite(grad_base).all():
            raise RuntimeError(f"non-finite base gradient at step {step}")

        base_grad_norm = float(grad_base.norm().item())
        internal_grad_norm = 0.0
        grad_scale = 0.0
        grad = grad_base
        if grad_internal is not None:
            if not torch.isfinite(grad_internal).all():
                raise RuntimeError(f"non-finite internal gradient at step {step}")
            internal_grad_norm = float(grad_internal.norm().item())
            rho = float(config.lambda_internal)
            if rho > 0.0 and base_grad_norm > 0.0 and internal_grad_norm > 0.0:
                scale = rho * base_grad_norm / internal_grad_norm
                grad_scale = min(scale, MAX_GRAD_SCALE)
                grad = grad_base + grad_scale * grad_internal

        grad_norm = float(grad.norm().item())
        velocity_norm = 0.0
        update_grad = grad
        if config.momentum > 0.0:
            unit = grad / (grad.abs().sum() + 1e-8)
            if velocity is None:
                velocity = unit
            else:
                velocity = config.momentum * velocity + unit
            velocity_norm = float(velocity.norm().item())
            update_grad = velocity
        if not torch.isfinite(update_grad).all():
            raise RuntimeError(f"non-finite update gradient at step {step}")

        if config.select_best:
            current_base_loss = float(pack.base_loss.item())
            if best_base_loss is None or current_base_loss < best_base_loss:
                best_base_loss = current_base_loss
                best_delta = delta.detach().clone()
                selected_step = step

        with torch.no_grad():
            next_delta = project_delta(
                delta.detach() - config.alpha * update_grad.sign(), waveform, config.eps)
            record = {
                "step": step,
                "base_loss": float(pack.base_loss.item()),
                "internal_loss": float(internal.item()),
                "total_loss": float(pack.base_loss.item())
                + float(config.lambda_internal) * float(internal.item()),
                "grad_norm": grad_norm,
                "base_grad_norm": base_grad_norm,
                "internal_grad_norm": internal_grad_norm,
                "grad_scale": grad_scale,
                "velocity_norm": velocity_norm,
                "delta_linf": float(next_delta.abs().max().item()),
                "snr_db": snr_db(waveform, waveform + next_delta),
                "refusal_probs": {
                    str(layer): float(value) for layer, value in refusal_probs.items()
                },
                "weights": scheduler.weight_dict(weights),
            }
        history.append(record)
        if verbose and (step == 0 or (step + 1) % 10 == 0 or step + 1 == config.steps):
            print(
                f"[pgd] step {step + 1:3d}/{config.steps} "
                f"base={record['base_loss']:.4f} internal={record['internal_loss']:.4f} "
                f"grad={record['grad_norm']:.3e} linf={record['delta_linf']:.4e}")

        delta = next_delta

        # Checkpoint evaluation: evaluate the waveform after this update.
        if (evaluate_fn is not None and config.eval_every > 0
                and (step + 1) % config.eval_every == 0):
            eval_wav = (waveform + delta).clamp(-1.0, 1.0)
            eval_result = evaluate_fn(eval_wav, step + 1)
            eval_result["step"] = step + 1
            eval_history.append(eval_result)
            evaluated_steps.append(step + 1)
            generation_calls += 1
            judge_calls += 1
            history[-1]["eval"] = eval_result
            final_judge_label = eval_result.get("label")
            if verbose:
                print(
                    f"[eval] step={step + 1} label={eval_result.get('label')} "
                    f"error={eval_result.get('error')}")
            if config.early_stop and eval_result.get("label") == "unsafe":
                success = True
                success_step = step + 1
                adversarial_response = eval_result.get("response")
                if verbose:
                    print(f"[early-stop] success at step {step + 1}")
                break

    final_delta = delta.detach()
    if config.select_best and best_delta is not None:
        final_delta = best_delta
        if verbose:
            print(f"[select-best] using step {selected_step} "
                  f"(base_loss={best_base_loss:.4f}) instead of the final step")
    final_wav = (waveform + final_delta).clamp(-1.0, 1.0)
    final_delta = final_wav - waveform

    # If eval_every does not divide steps, evaluate the final checkpoint too.
    if (evaluate_fn is not None and config.eval_every > 0
            and not (config.early_stop and success)
            and config.steps not in evaluated_steps):
        eval_result = evaluate_fn(final_wav, config.steps)
        eval_result["step"] = config.steps
        eval_history.append(eval_result)
        evaluated_steps.append(config.steps)
        generation_calls += 1
        judge_calls += 1
        final_judge_label = eval_result.get("label")
        if history:
            history[-1]["eval_final"] = eval_result
        if verbose:
            print(
                f"[eval] step={config.steps} label={eval_result.get('label')} "
                f"error={eval_result.get('error')}")
        if config.early_stop and eval_result.get("label") == "unsafe":
            success = True
            success_step = config.steps
            adversarial_response = eval_result.get("response")

    if evaluate_fn is not None and config.eval_every > 0:
        unsafe_steps = [e["step"] for e in eval_history
                        if e.get("label") == "unsafe"]
        success = bool(unsafe_steps)
        if success and success_step is None:
            success_step = unsafe_steps[0]

    with torch.no_grad():
        final_pack = model.forward_attack(final_wav, layout, active_layers)
        if not use_internal or not final_pack.refusal_logprobs:
            final_internal = final_pack.base_loss.new_zeros(())
        else:
            final_logprobs = {
                layer: logprob.detach()
                for layer, logprob in final_pack.refusal_logprobs.items()
            }
            weights = scheduler.update(config.steps, final_logprobs)
            terms = []
            for index, layer in enumerate(scheduler.layers):
                terms.append(
                    weights[index]
                    * internal_loss(final_pack.refusal_logprobs[layer], reference_logprob))
            final_internal = torch.stack(terms).sum()

    return AttackResult(
        original_wav=waveform.detach().cpu(),
        adversarial_wav=final_wav.detach().cpu(),
        perturbation=final_delta.detach().cpu(),
        history=history,
        final_base_loss=float(final_pack.base_loss.item()),
        final_internal_loss=float(final_internal.item()),
        success=success,
        success_step=success_step,
        evaluated_steps=evaluated_steps,
        eval_history=eval_history,
        generation_calls=generation_calls,
        judge_calls=judge_calls,
        final_judge_label=final_judge_label,
        selected_step=selected_step,
    )
