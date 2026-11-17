"""Output loss and internal refusal-logit-lens loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    kappa: float = 5.0,
    early_weight: float = 5.0,
) -> torch.Tensor:
    """CW-style margin loss over the supervised target tokens.

    Mirrors the WhisperInject Phi wrapper: only target tokens (labels != -100)
    are used, and the first three target tokens receive extra weight.
    """
    logits = logits.float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != -100
    if not valid_mask.any():
        return logits.sum() * 0.0

    positions = valid_mask[0].nonzero(as_tuple=True)[0]
    valid_logits = shift_logits[0, positions]
    valid_labels = shift_labels[0, positions].to(valid_logits.device)

    target_logits = valid_logits.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
    label_mask = torch.ones_like(valid_logits, dtype=torch.bool)
    label_mask.scatter_(1, valid_labels.unsqueeze(1), False)
    top_other = valid_logits.masked_fill(~label_mask, float("-inf")).max(dim=-1).values
    losses = torch.clamp(top_other - target_logits + kappa, min=0.0)

    weights = torch.ones_like(losses)
    weights[: min(3, losses.numel())] = early_weight
    return (losses * weights).sum() / weights.sum()


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over supervised target tokens."""
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def refusal_logprob(logits: torch.Tensor, refusal_ids: torch.Tensor) -> torch.Tensor:
    """Log probability mass of refusal first-tokens at one logit-lens readout."""
    logits = logits.float()
    refusal_ids = refusal_ids.to(logits.device)
    return (
        torch.logsumexp(logits[refusal_ids], dim=-1)
        - torch.logsumexp(logits, dim=-1)
    )


def internal_loss(logprob: torch.Tensor, reference_logprob: float) -> torch.Tensor:
    """Softplus penalty that only pushes layers still above the success reference.

    ``logprob`` is ``log P(refusal)``.  ``reference_logprob`` is ``log b`` where
    ``b`` is the refusal-probability reference (default 0.1).  A layer already
    below the reference receives a near-zero, still finite, penalty.
    """
    return F.softplus(logprob - reference_logprob)
