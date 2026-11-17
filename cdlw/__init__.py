"""Layer-adaptive waveform PGD on top of the AEGIS Qwen2-Audio runtime.

The package deliberately does not modify the AEGIS defense code.  It reuses the
AEGIS model adapter / prompt protocol / manifests and adds:

- an official-equivalent differentiable Whisper log-mel frontend,
- a differentiable last-prompt-token readout at selected decoder layers,
- the AEGIS refusal logit-lens probe as an internal attack objective,
- a static / dynamic layer-weight scheduler,
- a shared sign-PGD loop for all layer-weight variants.

The first-round prior is the AEGIS conclusion for Qwen2-Audio: the late decoder
layers 19-31 are the effective refusal-intervention range.  This code is an
attack experiment, not a causal-effect claim.
"""

__version__ = "0.1.0"
