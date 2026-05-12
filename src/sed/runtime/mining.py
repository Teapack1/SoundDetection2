"""Save live detections as trainable mined library clips."""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf

from sed.config import Config, ProjectLayout, resolve_active_model_dir
from sed.data.library import Library, MINED_CATEGORY
from sed.runtime.infer import RuntimeDetection
from sed.util import unique_path

log = logging.getLogger(__name__)


class DetectionMiner:
    """Persist selected runtime detections as first-class mined library clips."""

    def __init__(
        self,
        cfg: Config,
        project_id: str,
        classes: set[str],
        output_dir: str | None = None,
    ):
        self.cfg = cfg
        self.layout = ProjectLayout(project_id)
        self.project_id = project_id
        self.classes = set(classes)
        self.library = Library()
        self.root = self.library.root / (output_dir or MINED_CATEGORY)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"inference-{stamp}-{_uuid.uuid4().hex[:6]}"
        self.count = 0

    def maybe_save(self, det: RuntimeDetection) -> Path | None:
        if det.class_name not in self.classes:
            return None

        out_dir = self.root / _safe_name(det.class_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_uuid = _uuid.uuid4().hex
        self.count += 1
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = f"{stamp}_{_safe_name(det.class_name)}_{det.confidence:.3f}_{det.kind}_{clip_uuid[:8]}"
        wav_path = unique_path(out_dir / f"{stem}.wav")
        sf.write(str(wav_path), det.audio, det.sample_rate, subtype="PCM_16")

        now_iso = datetime.now(UTC).isoformat()
        mic_types = list(self.cfg.audio.microphone_type or [])
        best = det.best_channel
        mic_label = (
            str(mic_types[int(best)] or "")
            if best is not None and 0 <= int(best) < len(mic_types) else ""
        )
        clip_kind = "mono" if len(det.active_channels) <= 1 else "multichannel"
        sidecar = {
            "schema_version": 2,
            "category": MINED_CATEGORY,
            "label": det.class_name,
            "class_name": det.class_name,
            "project_id": self.project_id,
            "density": "normal",
            "review_status": "accepted",
            "uuid": clip_uuid,
            "recording_counter": self.count,
            "session_id": self.session_id,
            "origin": "runtime_mining",
            "created_at": now_iso,
            "recorded_at": now_iso,
            "predicted_class": det.class_name,
            "confidence": det.confidence,
            "kind": det.kind,
            "sample_rate": det.sample_rate,
            "channels": len(det.active_channels),
            "channel_count": len(det.active_channels),
            "channel_index": det.best_channel,
            "clip_kind": clip_kind,
            "active_channels": det.active_channels,
            "recorded_channels": list(det.active_channels),
            "triggered_channels": [det.best_channel] if det.best_channel is not None else [],
            "best_channel": det.best_channel,
            "microphone": det.best_channel,
            "microphone_label": mic_label,
            "microphone_type": mic_types or "generic",
            "mic_position": mic_label,
            "device": self.cfg.deploy.record_device,
            "duration_s": len(det.audio) / det.sample_rate if det.sample_rate else 0.0,
            "peak_dbfs": det.peak_dbfs,
            "rms_dbfs": det.rms_dbfs,
            "min_dbfs": det.min_dbfs,
            "active_targets": det.active_targets,
            "model_dir": str(_active_model_dir(self.layout, self.cfg).resolve()),
            "active_model": self.cfg.deploy.active_model or None,
            "classes": list(self.cfg.model.classes),
            "probs": det.probs,
            "smoothed": det.smoothed,
            "probs_percent": _percent(det.probs),
            "smoothed_percent": _percent(det.smoothed),
            "tags": ["mined", det.kind, det.class_name],
            "description": f"runtime mined {det.class_name} ({det.kind})",
        }
        with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        self.library.append_sidecar(wav_path)
        # DEBUG, not INFO: the operator-facing DETECT line already reports the
        # trigger clip's path, and threshold-kind clips are background capture —
        # logging each one floods the console. They are counted in stats instead.
        log.debug("mined %s (%s) -> %s", det.class_name, det.kind, wav_path)
        return wav_path


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)




def _percent(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value) * 100.0, 3) for key, value in values.items()}


def _active_model_dir(layout: ProjectLayout, cfg: Config) -> Path:
    """Active model subfolder for provenance; falls back to models/ if unresolved."""
    try:
        return resolve_active_model_dir(layout, cfg.deploy.active_model)
    except FileNotFoundError:
        return layout.models_dir
