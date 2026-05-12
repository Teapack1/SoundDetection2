"""Post-training BatchNorm running-stats recalibration.

Fine-tuning regimes routinely leave BatchNorm running statistics out of sync
with the weights that were actually learned: frozen stages keep their upstream
(AudioSet) stats while trainable layers shift the activation distribution,
mixup feeds convex-combined inputs whose statistics production never sees, and
small-batch fine-tunes estimate noisy batch moments. Recalibration streams
clean training batches through the *frozen* weights with only BatchNorm layers
in train mode (momentum=None → exact cumulative averages), then writes a new
checkpoint alongside the original so calibration / auto-test / export can all
consume the refreshed statistics.

This deliberately overrides ``training.freeze_batchnorm``'s preserved stats:
the point of the pass is to replace stale moments with ones measured on this
project's data. Models without BatchNorm (pure transformers) are a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sed.config import Config, ProjectLayout

log = logging.getLogger(__name__)


def recalibrate_batchnorm(
    cfg: Config,
    project_id: str,
    *,
    checkpoint_path: Path,
    num_batches: int | None = None,
    dataset_dir: Path | None = None,
) -> dict:
    """Refresh BN running stats on train data; write ``<stem>_bnrecal.ckpt``.

    Returns ``{"checkpoint": Path|None, "bn_layers": int, "batches": int,
    "samples": int}``. ``checkpoint`` is None when the model has no BatchNorm
    layers (nothing to recalibrate) — callers keep using the source checkpoint.
    """
    import torch
    from torch.utils.data import DataLoader

    from sed.train.datamodule import SedDataset
    from sed.train.lit_module import SedLitModule, load_classes

    checkpoint_path = Path(checkpoint_path)
    model_dir = checkpoint_path.parent
    layout = ProjectLayout(project_id)
    if dataset_dir is None:
        run_dataset = model_dir / "dataset"
        dataset_dir = run_dataset if (run_dataset / "train.csv").exists() else layout.dataset_dir

    classes_dir = model_dir if (model_dir / "classes.json").exists() else dataset_dir
    classes = load_classes(classes_dir)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch.load` with `weights_only=False`.*",
        )
        lit = SedLitModule.load_from_checkpoint(
            str(checkpoint_path), cfg=cfg, classes=classes, map_location="cpu"
        )

    bn_layers = [
        m for m in lit.model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_layers:
        log.info("bn recalibration: %s has no BatchNorm layers — skipping", cfg.model.backbone)
        return {"checkpoint": None, "bn_layers": 0, "batches": 0, "samples": 0}

    # Everything frozen and in eval mode EXCEPT the BN layers, which collect
    # exact cumulative running averages (momentum=None) from a clean reset.
    lit.eval()
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
        m.train()

    # Clean training inputs: train.csv rows through the eval feature path
    # (train=False → no SpecAugment), shuffled so the estimate is not biased
    # by CSV ordering (augmented rows cluster at the tail).
    ds = SedDataset(dataset_dir / "train.csv", classes, cfg, layout=layout, train=False)
    if len(ds) == 0:
        raise RuntimeError(f"bn recalibration: no readable training rows in {dataset_dir}")
    generator = torch.Generator().manual_seed(int(cfg.training.seed))
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    target_batches = int(num_batches if num_batches is not None else cfg.training.bn_recalibrate_batches)

    seen_batches = 0
    seen_samples = 0
    with torch.no_grad():
        for x, _y in loader:
            lit(x)
            seen_batches += 1
            seen_samples += int(x.shape[0])
            if seen_batches >= target_batches:
                break

    # Preserve the full Lightning checkpoint structure so every downstream
    # loader (eval, export, CheckpointBackend) reads the new file unchanged.
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    raw["state_dict"] = lit.state_dict()
    out_path = checkpoint_path.with_name(checkpoint_path.stem + "_bnrecal.ckpt")
    torch.save(raw, str(out_path))
    log.info(
        "bn recalibration: %d BN layers refreshed on %d batches (%d clips) → %s",
        len(bn_layers), seen_batches, seen_samples, out_path.name,
    )
    return {
        "checkpoint": out_path,
        "bn_layers": len(bn_layers),
        "batches": seen_batches,
        "samples": seen_samples,
    }
