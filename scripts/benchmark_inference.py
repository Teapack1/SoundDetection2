"""Benchmark OpenVINO inference latency on the local CPU.

Usage:
    python scripts/benchmark_inference.py --project example_venue --warmup 30 --iters 200

Prints p50/p95/p99 in ms — the number that matters for the latency budget.
"""

from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np
import openvino as ov

from sed.config import ProjectLayout, load_config
from sed.runtime.infer import load_manifest, resolve_model_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    cfg = load_config(args.project)
    layout = ProjectLayout(args.project)
    manifest = load_manifest(layout)
    model_path = resolve_model_path(layout, manifest)

    core = ov.Core()
    cm = core.compile_model(
        core.read_model(str(model_path)),
        "CPU",
        config={
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": str(args.threads),
        },
    )
    req = cm.create_infer_request()

    # Build a representative input
    n_mels = manifest["n_mels"]
    n_frames = int(round(
        manifest["window_seconds"] * manifest["sample_rate"] / manifest["hop_length"]
    )) + 1
    x = np.random.randn(1, 1, n_mels, n_frames).astype(np.float32)

    # Warmup
    for _ in range(args.warmup):
        req.infer({0: x})

    # Time
    times_ms: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        req.infer({0: x})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    print(f"model:    {model_path.name}")
    print(f"shape:    1×1×{n_mels}×{n_frames}")
    print(f"threads:  {args.threads}")
    print(f"iters:    {n}")
    print(f"mean:     {mean(times_ms):6.2f} ms")
    print(f"p50:      {times_ms[n // 2]:6.2f} ms")
    print(f"p95:      {times_ms[int(n * 0.95)]:6.2f} ms")
    print(f"p99:      {times_ms[int(n * 0.99)]:6.2f} ms")
    print(f"min/max:  {times_ms[0]:6.2f} / {times_ms[-1]:6.2f} ms")


if __name__ == "__main__":
    main()
