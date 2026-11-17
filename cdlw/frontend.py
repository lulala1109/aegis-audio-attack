"""Official-equivalent differentiable Whisper log-mel frontend for Qwen2-Audio.

The feature values are identical to ``WhisperFeatureExtractor._torch_extract_fbank_features``
for a waveform padded to its native 30 s length.  The only deliberate difference
is that this implementation keeps the PyTorch graph alive so gradients can reach
the waveform.

Reference parameters are read from the fixed processor:
    n_fft=400, hop_length=160, feature_size=128, n_samples=480000,
    Hann window, center=True (reflect), last STFT frame dropped,
    power spectrogram, log10 with 1e-10 floor, 8.0 dynamic-range clamp,
    (x + 4) / 4.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class WhisperLogMel(torch.nn.Module):
    """Differentiable log-mel frontend matching the Qwen2-Audio processor."""

    def __init__(self, feature_extractor, device: str | torch.device = "cpu"):
        super().__init__()
        self.n_fft = int(feature_extractor.n_fft)
        self.hop_length = int(feature_extractor.hop_length)
        self.feature_size = int(feature_extractor.feature_size)
        self.n_samples = int(feature_extractor.n_samples)
        self.dither = float(getattr(feature_extractor, "dither", 0.0))

        mel = np.asarray(feature_extractor.mel_filters, dtype=np.float32)
        if mel.shape[0] != self.n_fft // 2 + 1 or mel.shape[1] != self.feature_size:
            raise ValueError(
                f"unexpected mel filter shape {mel.shape}; expected "
                f"{(self.n_fft // 2 + 1, self.feature_size)}")
        self.register_buffer("mel_filters", torch.from_numpy(mel), persistent=False)
        self.register_buffer(
            "window", torch.hann_window(self.n_fft, dtype=torch.float32), persistent=False)
        self.to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return ``[B, 128, 3000]`` log-mel features, differentiable in ``waveform``."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"waveform must be [B, T] or [T], got {tuple(waveform.shape)}")
        waveform = waveform.to(device=self.mel_filters.device, dtype=torch.float32)

        # The official extractor pads/truncates to n_samples before the STFT.
        length = waveform.shape[-1]
        if length < self.n_samples:
            waveform = F.pad(waveform, (0, self.n_samples - length))
        elif length > self.n_samples:
            waveform = waveform[..., : self.n_samples]

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        # _torch_extract_fbank_features drops the final frame unconditionally.
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if log_spec.dim() == 2:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        else:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        return (log_spec + 4.0) / 4.0


def official_features(processor, waveform: np.ndarray) -> torch.Tensor:
    """Run the fixed official feature extractor for parity checks."""
    fe = processor.feature_extractor
    out = fe(
        [np.asarray(waveform, dtype=np.float32)],
        sampling_rate=fe.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return out["input_features"]
