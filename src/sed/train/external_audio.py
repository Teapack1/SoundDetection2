"""Optional external audio-model adapters.

These wrappers are deliberately small and import optional packages lazily. The
curated catalog can list promising pretrained candidates without making a fresh
install fail at import time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _missing(package: str, install_hint: str) -> ImportError:
    return ImportError(
        f"{package} is required for this experimental external model. "
        f"Install it first, for example: {install_hint}"
    )


def _as_single_channel_feature(x: torch.Tensor, model_name: str) -> torch.Tensor:
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(f"{model_name} accepts only 1-channel audio features, got {x.shape[1]}")
        return x[:, 0]
    if x.dim() == 3:
        return x
    raise ValueError(f"{model_name} expects (B, 1, bands, frames) or (B, bands, frames), got {tuple(x.shape)}")


class CEDEncoder(nn.Module):
    """Shared CED-family encoder loaded from Hugging Face.

    The upstream model expects CED log-mel features shaped ``(B, 64, T)``. This
    wrapper accepts the training backend's tensor ``(B, 1, 64, T)`` and returns a
    pooled embedding so the local classifier head can train on top.
    """

    model_name = "mispeech/ced-tiny"

    def __init__(self, in_chans: int = 1, n_bands: int = 64, n_frames: int = 41,
                 pretrained: bool = True, model_name: str | None = None,
                 local_files_only: bool = False, trust_remote_code: bool = True,
                 **_ignored):
        super().__init__()
        _ = n_frames
        if int(in_chans) != 1:
            raise ValueError("CED is 1-channel only; select the CED 64-mel feature contract")
        if int(n_bands) != 64:
            raise ValueError(f"CED expects 64 mel bands, got {n_bands}")
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForAudioClassification
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise _missing("transformers", "pip install transformers accelerate") from exc

        common = dict(trust_remote_code=bool(trust_remote_code), local_files_only=bool(local_files_only))
        self.model_name = str(model_name or type(self).model_name)
        if pretrained:
            self.model = self._load_encoder(AutoModel, AutoModelForAudioClassification, self.model_name, common)
        else:
            config = AutoConfig.from_pretrained(self.model_name, **common)
            try:
                self.model = AutoModel.from_config(config, trust_remote_code=bool(trust_remote_code))
            except Exception:
                classifier = AutoModelForAudioClassification.from_config(
                    config, trust_remote_code=bool(trust_remote_code)
                )
                self.model = self._base_model(classifier)
        self.out_channels = int(getattr(getattr(self.model, "config", object()), "embed_dim", 192))

    @staticmethod
    def _base_model(model: nn.Module) -> nn.Module:
        # ``base_model`` last: PreTrainedModel.base_model falls back to the
        # model itself when the prefix attribute is absent (CED's prefix is
        # "ced" but its submodule is named "encoder"), which would return the
        # full classifier including its sigmoid head.
        for attr in ("ced", "audio_model", "encoder", "model", "base_model"):
            candidate = getattr(model, attr, None)
            if isinstance(candidate, nn.Module) and candidate is not model:
                return candidate
        return model

    @classmethod
    def _load_encoder(cls, auto_model, auto_classifier, model_name: str, common: dict) -> nn.Module:
        # The hub checkpoint stores CedForAudioClassification weights
        # (``encoder.*`` + ``outputlayer.*``) while AutoModel builds the bare
        # CedModel whose state dict has no ``encoder.`` prefix — from_pretrained
        # then matches nothing and silently returns a randomly initialized
        # encoder. Load through the classifier, whose keys match exactly, and
        # unwrap its encoder.
        try:
            return cls._base_model(auto_classifier.from_pretrained(model_name, **common))
        except Exception:
            return auto_model.from_pretrained(model_name, **common)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = _as_single_channel_feature(x, "CED")
        out = self.model(input_values=values)
        emb = getattr(out, "last_hidden_state", None)
        if emb is None:
            emb = getattr(out, "hidden_states", None)
            if isinstance(emb, (tuple, list)) and emb:
                emb = emb[-1]
        if emb is None:
            emb = getattr(out, "logits", None)
        if emb is None:
            raise RuntimeError("CED output did not expose hidden states or logits")
        if emb.dim() == 3:
            return emb.mean(dim=1)
        return emb


class CEDTinyEncoder(CEDEncoder):
    model_name = "mispeech/ced-tiny"


class CEDMiniEncoder(CEDEncoder):
    model_name = "mispeech/ced-mini"


class CEDSmallEncoder(CEDEncoder):
    model_name = "mispeech/ced-small"


class CEDBaseEncoder(CEDEncoder):
    model_name = "mispeech/ced-base"


__all__ = [
    "CEDEncoder",
    "CEDTinyEncoder",
    "CEDMiniEncoder",
    "CEDSmallEncoder",
    "CEDBaseEncoder",
]
