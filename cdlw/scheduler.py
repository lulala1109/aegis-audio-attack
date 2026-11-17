"""Layer-weight schedulers.

The first-round prior comes from AEGIS's Qwen2-Audio setting:
``LATE_FROM=19`` with 32 decoder layers, i.e. the key late-layer set 19-31.
This module does not claim that those weights are causal effects.  It only
tests whether allocating the internal objective by AEGIS's late-layer range,
and adapting that allocation to the current refusal state, helps the waveform
PGD.

Modes
-----
none      : no internal objective (plain PGD)
uniform   : equal weight over the configured key layers
static    : fixed AEGIS key-layer prior
dynamic   : logit-space softmax focus on the current refusal signal

Dynamic mode reweights the layers with a softmax over their (EMA-smoothed)
refusal log-probability relative to the reference.  Working in logit space
makes the weights invariant to common-mode shifts of the refusal signal:
whether the whole model is refusing or the whole model is broken, only the
*relative* differences between layers change the focus.  ``temperature`` is
the softmax temperature in logit space (smaller = harder focus on the worst
layer).  ``eta`` is kept for backward compatibility and is unused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class SchedulerConfig:
    mode: str = "dynamic"
    temperature: float = 1.0
    eta: float = 0.2
    beta: float = 0.5
    update_every: int = 1
    reference_prob: float = 0.02


class LayerScheduler:
    """Maintain one normalized weight per configured decoder layer."""

    VALID_MODES = ("none", "uniform", "static", "dynamic")

    def __init__(
        self,
        layers: list[int],
        mode: str = "dynamic",
        prior: dict[int, float] | None = None,
        config: SchedulerConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown scheduler mode {mode!r}; expected {self.VALID_MODES}")
        self.mode = mode
        self.layers = [int(layer) for layer in layers]
        self.config = config or SchedulerConfig()
        self.device = device

        if mode != "none" and not self.layers:
            raise ValueError(f"mode {mode!r} requires at least one layer")

        if prior is None:
            values = torch.ones(len(self.layers), dtype=torch.float32)
        else:
            values = torch.tensor(
                [float(prior.get(layer, 0.0)) for layer in self.layers],
                dtype=torch.float32,
            )
            if (values < 0).any():
                raise ValueError("prior weights must be nonnegative")
            if values.sum() <= 0:
                raise ValueError("prior weights must have a positive sum")

        self.c = (values / values.sum()).to(device)
        self.reference_logprob = math.log(max(float(self.config.reference_prob), 1e-8))
        self.weights = self.c.clone()
        self._logits: torch.Tensor | None = None
        self._started = False

    def reset(self) -> None:
        self.weights = self.c.clone()
        self._logits = None
        self._started = False

    def update(
        self, step: int, refusal_logprobs: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Return the weights to use for the current loss.

        ``refusal_logprobs`` are detached per-layer refusal log-probabilities
        from the current waveform.  They must not enter the attack graph.
        """
        if self.mode == "none":
            return torch.zeros(0, device=self.device)
        if self.mode in ("uniform", "static"):
            return self.weights

        if not self._started:
            self._started = True
            return self.weights

        if self.config.update_every <= 0 or step % self.config.update_every != 0:
            return self.weights

        logprob = torch.stack(
            [refusal_logprobs[layer].detach().float().to(self.device)
             for layer in self.layers]
        )
        if self._logits is None:
            self._logits = logprob
        else:
            beta = float(self.config.beta)
            self._logits = beta * self._logits + (1.0 - beta) * logprob

        temperature = max(float(self.config.temperature), 1e-3)
        logits = (self._logits - self.reference_logprob) / temperature + self.c.log()
        self.weights = torch.softmax(logits, dim=0)
        return self.weights

    def weight_dict(self, weights: torch.Tensor | None = None) -> dict[int, float]:
        if self.mode == "none":
            return {}
        w = self.weights if weights is None else weights
        return {layer: float(v) for layer, v in zip(self.layers, w.detach().cpu())}

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "layers": self.layers,
            "prior": [float(x) for x in self.c.detach().cpu()],
            "weights": [float(x) for x in self.weights.detach().cpu()],
            "config": self.config.__dict__,
        }


def key_layer_prior(layers: list[int], scheme: str = "uniform") -> dict[int, float]:
    """Fixed prior over the AEGIS key layers.

    ``uniform`` is the neutral, defensible first-round choice.  ``late_decay``
    is an optional stress test that gives later layers more weight; it is not
    an AEGIS result and should be reported as an ablation.
    """
    if scheme == "uniform":
        return {layer: 1.0 for layer in layers}
    if scheme == "late_decay":
        lo, hi = min(layers), max(layers)
        span = max(hi - lo, 1)
        return {layer: float(layer - lo + 1) / span for layer in layers}
    raise ValueError(f"unknown prior scheme {scheme!r}")
