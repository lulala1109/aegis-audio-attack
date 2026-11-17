"""Differentiable Qwen2-Audio attack wrapper built on the AEGIS adapter."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
import torch

from .bridge import configure_offline, get_adapter
from .frontend import WhisperLogMel
from .objectives import margin_loss, refusal_logprob


@dataclass
class Layout:
    """Fixed token/feature layout for one (audio, target) pair."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    feature_attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_len: int
    target_text: str
    prompt_text: str
    official_input_features: torch.Tensor


@dataclass
class ForwardPack:
    base_loss: torch.Tensor
    refusal_logprobs: dict[int, torch.Tensor]
    layer_states: dict[int, torch.Tensor]
    prompt_len: int


def _first_ids(tokenizer, phrases) -> list[int]:
    ids = set()
    for phrase in phrases:
        for variant in (phrase, " " + phrase):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[0]))
    return sorted(ids)


class QwenAudioAttackModel:
    """One loaded Qwen2-Audio model plus differentiable attack helpers."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        prompt_mode: str = "native",
        hf_home: str | None = None,
        max_seconds: float = 30.0,
        attn_implementation: str = "eager",
    ):
        configure_offline(hf_home)
        self.device = torch.device(device)
        self.prompt_mode = prompt_mode
        self.max_seconds = float(max_seconds)

        self.adapter = get_adapter("qwen2_audio")
        self.adapter.model_id = model_path
        self.adapter.set_prompt_mode(prompt_mode)

        # Load Qwen directly so attention can be forced to eager.  The default
        # SDPA/Flash backward is not deterministic under torch.use_deterministic_algorithms.
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.tokenizer = self.adapter.tokenizer(self.processor)
        self.sample_rate = int(self.adapter.sampling_rate(self.processor))
        self.layers = self.adapter.decoder_layers(self.model)
        self.final_norm = self.adapter.final_norm(self.model)
        self.lm_head = self.adapter.lm_head(self.model)
        self.num_layers = len(self.layers)

        self.frontend = WhisperLogMel(
            self.processor.feature_extractor, device=self.device)
        self.refusal_ids = torch.tensor(
            _first_ids(self.tokenizer, self.adapter.refusal_phrases),
            dtype=torch.long,
            device=self.device,
        )
        if self.refusal_ids.numel() == 0:
            raise RuntimeError("no refusal first-token ids were found")

        self._capture: dict[int, torch.Tensor] = {}
        self._capture_layers: set[int] = set()
        self._capture_prompt_len = 0
        self._capture_enabled = False
        for layer in range(self.num_layers):
            self.layers[layer].register_forward_hook(self._make_hook(layer))

    # ------------------------------------------------------------------
    # audio / layout
    # ------------------------------------------------------------------
    def load_wav(self, path: str) -> torch.Tensor:
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        if self.max_seconds > 0:
            wav = wav[: int(self.max_seconds * self.sample_rate)]
        if wav.size == 0:
            raise ValueError(f"empty audio: {path}")
        # Keep the fixed attack reference inside the legal waveform range.
        wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        return torch.from_numpy(wav).unsqueeze(0)

    def build_layout(
        self,
        waveform: np.ndarray | torch.Tensor,
        target_text: str,
        append_eos: bool = True,
    ) -> Layout:
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.detach().cpu().squeeze(0).numpy()
        else:
            waveform_np = np.asarray(waveform, dtype=np.float32)

        content = self.adapter.conversation_content("dummy.wav", n_audio=1)
        conversation = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)

        prompt = self.processor(
            text=prompt_text,
            audio=[waveform_np],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        prompt_ids = prompt["input_ids"]
        prompt_len = int(prompt_ids.shape[1])

        target_ids = self.tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt").input_ids
        if append_eos and self.tokenizer.eos_token_id is not None:
            eos = torch.tensor(
                [[int(self.tokenizer.eos_token_id)]], dtype=target_ids.dtype)
            target_ids = torch.cat([target_ids, eos], dim=1)

        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        return Layout(
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_attention_mask=prompt["feature_attention_mask"],
            labels=labels,
            prompt_len=prompt_len,
            target_text=target_text,
            prompt_text=prompt_text,
            official_input_features=prompt["input_features"],
        )

    # ------------------------------------------------------------------
    # differentiable forward
    # ------------------------------------------------------------------
    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if not self._capture_enabled or layer not in self._capture_layers:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            position = self._capture_prompt_len - 1
            if hidden.shape[1] > position:
                self._capture[layer] = hidden[:, position, :]
        return hook

    def _logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden is [D] or [1, D] in the model dtype.  Keep the graph; float() is safe.
        hidden = hidden.unsqueeze(0) if hidden.dim() == 1 else hidden
        return self.lm_head(self.final_norm(hidden))[0].float()

    def forward_attack(
        self,
        waveform: torch.Tensor,
        layout: Layout,
        layers: list[int],
    ) -> ForwardPack:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device, dtype=torch.float32)

        self._capture = {}
        self._capture_layers = set(int(layer) for layer in layers)
        self._capture_prompt_len = int(layout.prompt_len)
        self._capture_enabled = True
        try:
            features = self.frontend(waveform)
            outputs = self.model(
                input_ids=layout.input_ids.to(self.device),
                input_features=features,
                attention_mask=layout.attention_mask.to(self.device),
                feature_attention_mask=layout.feature_attention_mask.to(self.device),
                use_cache=False,
                return_dict=True,
            )
        finally:
            self._capture_enabled = False

        base_loss = margin_loss(
            outputs.logits, layout.labels.to(self.device), kappa=5.0, early_weight=5.0)
        refusal_logprobs = {}
        layer_states = {}
        for layer, hidden in self._capture.items():
            layer_states[layer] = hidden
            lens_logits = self._logit_lens(hidden)
            refusal_logprobs[layer] = refusal_logprob(lens_logits, self.refusal_ids)
        return ForwardPack(
            base_loss=base_loss,
            refusal_logprobs=refusal_logprobs,
            layer_states=layer_states,
            prompt_len=int(layout.prompt_len),
        )

    # ------------------------------------------------------------------
    # official generation
    # ------------------------------------------------------------------
    def generate_from_path(self, path: str, max_new_tokens: int = 256) -> str:
        inputs = self.adapter.build_inputs(
            self.model, self.processor, path, str(self.device))
        with torch.inference_mode():
            new_ids = self.adapter.generate(self.model, inputs, max_new_tokens)
        return self.adapter.decode(self.processor, new_ids)

    def generate_from_tensor(
        self, waveform: torch.Tensor, max_new_tokens: int = 256) -> str:
        array = waveform.detach().cpu().squeeze(0).numpy().astype(np.float32)
        fd, path = tempfile.mkstemp(prefix="cdlw_", suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, array, self.sample_rate, subtype="FLOAT")
            return self.generate_from_path(path, max_new_tokens=max_new_tokens)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def reference_logprob(self, probability: float) -> float:
        return math.log(max(float(probability), 1e-8))
