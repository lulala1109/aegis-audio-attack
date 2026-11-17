"""CPU-only checks for projection and layer-weight scheduling."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdlw.pgd import project_delta  # noqa: E402
from cdlw.scheduler import LayerScheduler, SchedulerConfig  # noqa: E402


def test_projection_respects_linf_and_range():
    x = torch.tensor([[-0.99, 0.0, 0.99]])
    delta = torch.tensor([[0.5, 0.5, 0.5]])
    projected = project_delta(delta, x, eps=0.01)
    assert projected.abs().max().item() <= 0.01 + 1e-7
    assert ((x + projected).abs() <= 1.0 + 1e-7).all()


def test_dynamic_weights_are_normalized_and_reset():
    layers = [19, 20, 21]
    config = SchedulerConfig(temperature=1.0, eta=0.2, beta=0.5, update_every=1)
    scheduler = LayerScheduler(layers, mode="dynamic", config=config, device="cpu")
    probs = {
        19: torch.tensor(0.9),
        20: torch.tensor(0.5),
        21: torch.tensor(0.01),
    }
    logprobs = {layer: prob.clamp_min(1e-12).log() for layer, prob in probs.items()}
    scheduler.update(0, logprobs)
    for step in range(1, 7):
        weights = scheduler.update(step, logprobs)
        assert torch.isfinite(weights).all()
        assert abs(float(weights.sum()) - 1.0) < 1e-5
        assert (weights >= 0).all()
    assert solver_weights_differ(scheduler.weights, scheduler.c)
    scheduler.reset()
    assert torch.allclose(scheduler.weights, scheduler.c)


def solver_weights_differ(a, b):
    return not torch.allclose(a, b, atol=1e-6)


def test_dynamic_focuses_worst_layer():
    layers = [19, 20, 21]
    config = SchedulerConfig(temperature=0.1, beta=0.0, update_every=1)
    scheduler = LayerScheduler(layers, mode="dynamic", config=config, device="cpu")
    logprobs = {19: torch.tensor(-1.0), 20: torch.tensor(-2.0), 21: torch.tensor(-5.0)}
    scheduler.update(0, logprobs)
    weights = scheduler.update(1, logprobs)
    assert float(weights[0]) > float(weights[1]) > float(weights[2])


def test_dynamic_weights_are_shift_invariant():
    layers = [19, 20, 21]
    config = SchedulerConfig(temperature=1.0, beta=0.0, update_every=1)
    scheduler = LayerScheduler(layers, mode="dynamic", config=config, device="cpu")
    logprobs = {19: torch.tensor(0.0), 20: torch.tensor(-1.0), 21: torch.tensor(-3.0)}
    scheduler.update(0, logprobs)
    weights = scheduler.update(1, logprobs).clone()
    shifted = {layer: value - 5.0 for layer, value in logprobs.items()}
    scheduler.update(2, shifted)
    weights_shifted = scheduler.update(3, shifted)
    assert torch.allclose(weights, weights_shifted, atol=1e-5)


def test_static_does_not_change():
    layers = [19, 20]
    scheduler = LayerScheduler(layers, mode="static", device="cpu")
    logprobs = {19: torch.tensor(-0.1), 20: torch.tensor(-4.6)}
    before = scheduler.update(0, logprobs).clone()
    for step in range(1, 10):
        after = scheduler.update(step, logprobs)
        assert torch.allclose(before, after)
