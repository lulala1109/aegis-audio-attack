"""Differentiable audio-attack wrappers built on the AEGIS adapters.

``QwenAudioAttackModel`` is the original Qwen2-Audio path and is unchanged.
``Phi4AudioAttackModel`` adds the Phi-4-multimodal path behind the same
interface; ``build_attack_model`` selects between them so callers stay generic.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from inspect import signature

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
    # Phi-4 carries the official audio token count and input mode alongside the
    # processor-built ids; Qwen2-Audio leaves these at None.
    audio_embed_sizes: torch.Tensor | None = None
    input_mode: torch.Tensor | None = None


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


class Phi4LogFbank(torch.nn.Module):
    """Differentiable log-filterbank frontend matching the Phi-4 processor.

    Reproduces ``Phi4MMAudioFeatureExtractor._extract_features`` for 16 kHz mono
    audio (pre-emphasis, 400/160 Hamming-windowed rFFT of size 512, SpeechLib
    mel bank, log with a 1.0 floor).  Values match the official numpy path to
    ~1e-6 while keeping the graph alive so gradients reach the waveform.
    """

    WIN_LENGTH = 400
    HOP_LENGTH = 160
    N_FFT = 512
    PREEMPHASIS = 0.97
    SCALE = 32768.0

    def __init__(self, feature_extractor, device: str | torch.device = "cpu"):
        super().__init__()
        self.feature_size = int(feature_extractor.feature_size)
        self.compression_rate = int(getattr(feature_extractor, "compression_rate", 8))
        self.qformer_compression_rate = int(
            getattr(feature_extractor, "qformer_compression_rate", 1))

        mel = np.asarray(feature_extractor._mel, dtype=np.float32)
        expected = (self.N_FFT // 2 + 1, self.feature_size)
        if mel.shape != expected:
            raise ValueError(
                f"unexpected mel filter shape {mel.shape}; expected {expected}")
        self.register_buffer("mel_filters", torch.from_numpy(mel), persistent=False)
        # np.hamming(400) is the symmetric window; float64 matches the extractor.
        self.register_buffer(
            "window",
            torch.hamming_window(self.WIN_LENGTH, periodic=False, dtype=torch.float64),
            persistent=False,
        )
        self.to(device)

    def num_frames(self, n_samples: int) -> int:
        if n_samples < self.WIN_LENGTH:
            return 0
        return (int(n_samples) - self.WIN_LENGTH) // self.HOP_LENGTH + 1

    def embed_size(self, n_frames: int) -> int:
        def ceil_div(value: int, divisor: int) -> int:
            return -(-int(value) // int(divisor))

        return ceil_div(
            ceil_div(n_frames, self.compression_rate), self.qformer_compression_rate)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return ``[B, T, 80]`` log-fbank features, differentiable in waveform."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"waveform must be [B, T] or [T], got {tuple(waveform.shape)}")
        waveform = waveform.to(
            device=self.mel_filters.device, dtype=torch.float32)
        if waveform.shape[-1] < self.WIN_LENGTH:
            raise ValueError(
                f"waveform has {waveform.shape[-1]} samples; "
                f"Phi-4 needs at least {self.WIN_LENGTH}")

        frames = waveform.unfold(-1, self.WIN_LENGTH, self.HOP_LENGTH)
        # The official extractor rolls each frame by one and then copies the
        # second sample into the first column (not the previous frame's tail).
        previous = torch.cat([frames[..., :1], frames[..., :-1]], dim=-1)
        emphasized = (frames - self.PREEMPHASIS * previous) * self.SCALE

        spectrum = torch.fft.rfft(
            emphasized.to(torch.float64) * self.window, n=self.N_FFT, dim=-1)
        magnitude = spectrum.to(torch.complex64).abs()
        power = magnitude ** 2
        return torch.clamp(power @ self.mel_filters, min=1.0).log()


class Phi4AudioAttackModel:
    """One loaded Phi-4-multimodal model plus differentiable attack helpers."""

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

        self.adapter = get_adapter("phi4_mm")
        self.adapter.model_id = model_path
        self.adapter.set_prompt_mode(prompt_mode)

        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            GenerationConfig,
        )
        from transformers.modeling_utils import PreTrainedModel

        from .bridge import registry_module

        registry = registry_module()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)
        # The published config pins flash_attention_2; the attack needs eager so
        # the waveform gradient is deterministic and flash-attn is not required.
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)
        config._attn_implementation = attn_implementation
        # The speech encoder checkpoints every conformer block and defaults to
        # the reentrant implementation, which forbids torch.autograd.grad().
        # Non-reentrant checkpointing keeps the same memory profile and works
        # with the PGD loop.
        audio_config = getattr(config, "audio_processor", None)
        if isinstance(audio_config, dict):
            encoder_config = audio_config.get("config") or {}
            activation = encoder_config.get("activation_checkpointing")
            if isinstance(activation, dict):
                activation["reentrant"] = False
        # Phi-4's remote attention still calls the pre-4.57 cache API.
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length, layer_idx=0):
                max_length = self.get_max_cache_shape()
                previous_seq_length = self.get_seq_length(layer_idx)
                if (max_length is not None
                        and previous_seq_length + new_seq_length > max_length):
                    return max_length - new_seq_length
                return previous_seq_length

            DynamicCache.get_usable_length = _get_usable_length
        # Eager attention keeps the backward pass deterministic, matching the
        # Qwen path; Phi-4 is loaded on CPU first because its remote code is
        # incompatible with meta-device initialization on recent Transformers.
        # The registry hook only targets Transformers >= 5; on 4.x its
        # get_init_context has a different signature and must be left alone.
        get_init = PreTrainedModel.__dict__.get("get_init_context")
        needs_cpu_context = (
            get_init is not None
            and "allow_all_kernels" in signature(get_init.__func__).parameters
        )
        context = (
            registry.phi_cpu_init_context(PreTrainedModel)
            if needs_cpu_context else nullcontext()
        )
        with context:
            # peft wraps the inner Phi4MMModel and unconditionally copies its
            # prepare_inputs_for_generation, which only exists on Transformers>=5.
            # Supply the standard GenerationMixin implementation on 4.x.
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from transformers.generation.utils import GenerationMixin

            outer_cls = get_class_from_dynamic_module(
                "modeling_phi4mm.Phi4MMForCausalLM",
                model_path,
                local_files_only=True,
            )
            inner_cls = getattr(sys.modules[outer_cls.__module__], "Phi4MMModel", None)
            if (inner_cls is not None
                    and not hasattr(inner_cls, "prepare_inputs_for_generation")):
                inner_cls.prepare_inputs_for_generation = (
                    GenerationMixin.prepare_inputs_for_generation)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
            )
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model._portable_gcfg = GenerationConfig.from_pretrained(
            model_path, local_files_only=True)

        self.tokenizer = self.adapter.tokenizer(self.processor)
        self.sample_rate = int(self.adapter.sampling_rate(self.processor))
        self.layers = self.adapter.decoder_layers(self.model)
        self.final_norm = self.adapter.final_norm(self.model)
        self.lm_head = self.adapter.lm_head(self.model)
        self.num_layers = len(self.layers)

        self.frontend = Phi4LogFbank(
            self.processor.audio_processor, device=self.device)
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
        wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        return torch.from_numpy(wav).unsqueeze(0)

    def _prompt_text(self) -> str:
        return (
            f"<|user|><|audio_1|>{self.adapter.effective_prompt()}"
            f"<|end|><|assistant|>"
        )

    def _target_ids(self, target_text: str, append_eos: bool) -> torch.Tensor:
        target_ids = self.tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt").input_ids
        if append_eos:
            end_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
            if end_id is None or end_id == self.tokenizer.unk_token_id:
                end_id = self.tokenizer.eos_token_id
            if end_id is not None:
                eos = torch.tensor(
                    [[int(end_id)]], dtype=target_ids.dtype)
                target_ids = torch.cat([target_ids, eos], dim=1)
        return target_ids

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

        prompt_text = self._prompt_text()
        prompt = self.processor(
            text=prompt_text,
            audios=[(waveform_np, self.sample_rate)],
            return_tensors="pt",
        )
        prompt_ids = prompt["input_ids"]
        prompt_len = int(prompt_ids.shape[1])

        official_audio = prompt["input_audio_embeds"]
        n_frames = self.frontend.num_frames(len(waveform_np))
        if official_audio.shape[1] != n_frames:
            raise RuntimeError(
                f"frontend frame mismatch: official {official_audio.shape[1]} "
                f"vs differentiable {n_frames}")
        expected_embed = self.frontend.embed_size(n_frames)
        if int(prompt["audio_embed_sizes"].sum().item()) != expected_embed:
            raise RuntimeError(
                f"audio token mismatch: processor "
                f"{int(prompt['audio_embed_sizes'].sum().item())} vs frontend "
                f"{expected_embed}")

        target_ids = self._target_ids(target_text, append_eos)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        return Layout(
            input_ids=input_ids,
            attention_mask=attention_mask,
            feature_attention_mask=None,
            labels=labels,
            prompt_len=prompt_len,
            target_text=target_text,
            prompt_text=prompt_text,
            official_input_features=official_audio,
            audio_embed_sizes=prompt["audio_embed_sizes"],
            input_mode=prompt["input_mode"],
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
        if layout.audio_embed_sizes is None or layout.input_mode is None:
            raise RuntimeError("Phi-4 layout is missing audio_embed_sizes/input_mode")

        self._capture = {}
        self._capture_layers = set(int(layer) for layer in layers)
        self._capture_prompt_len = int(layout.prompt_len)
        self._capture_enabled = True
        try:
            features = self.frontend(waveform)
            if features.shape[1] != layout.official_input_features.shape[1]:
                raise RuntimeError(
                    f"differentiable features have {features.shape[1]} frames; "
                    f"layout expects {layout.official_input_features.shape[1]}")
            outputs = self.model(
                input_ids=layout.input_ids.to(self.device),
                attention_mask=layout.attention_mask.to(self.device),
                input_audio_embeds=features,
                audio_embed_sizes=layout.audio_embed_sizes.to(self.device),
                audio_attention_mask=None,
                input_mode=layout.input_mode.to(self.device),
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


AttackModel = QwenAudioAttackModel | Phi4AudioAttackModel

ATTACK_MODELS = {
    "qwen2_audio": QwenAudioAttackModel,
    "phi4_mm": Phi4AudioAttackModel,
}


def build_attack_model(model_type: str = "qwen2_audio", **kwargs) -> AttackModel:
    """Instantiate the attack wrapper for ``model_type``."""
    if model_type not in ATTACK_MODELS:
        raise ValueError(
            f"unknown attack model {model_type!r}; "
            f"choose from {sorted(ATTACK_MODELS)}")
    return ATTACK_MODELS[model_type](**kwargs)
