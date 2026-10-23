#!/usr/bin/env python3
"""Train an NVIDIA pairwise-distance TCNN variant WITHOUT touching the shipped checkpoint.

The trainer's CLI has no --output-dir and writes to output/landmark_2d_106/nvidia_tcnn/
(overwriting best_model.pt -- this destroyed the checkpoint once). This wrapper sets
output_dir to output/landmark_2d_106/variants/<tag>/ and refuses to run otherwise.
Architecture is left unchanged (no embed-BN) so the fusion export/val-sims code, which
rebuilds the default architecture, can load the result.

    .venv/Scripts/python.exe scripts/train_nvidia_variant.py --tag nv_t01_s1 --temperature 0.1 \
        --iterations 6000 --seed 1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "output/landmark_2d_106/nvidia_tcnn"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--iterations", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    # Imported under its real module name: Windows DataLoader workers are spawned and must be
    # able to re-import the dataset classes by module path (a custom spec name breaks pickling).
    sys.path.insert(0, str(ROOT / "models/landmarks/nvidia/src"))
    import trainer as st  # noqa: E402
    out = ROOT / "output/landmark_2d_106/variants" / a.tag
    if out.resolve() == SHIPPED.resolve():
        raise SystemExit("refusing to write into the shipped checkpoint folder")
    o = {"output_dir": out, "iterations": a.iterations, "seed": a.seed}
    if a.temperature is not None:
        o["temperature"] = a.temperature
    if a.lr is not None:
        o["learning_rate"] = a.lr
    cfg = replace(st.CONFIG, **o)
    out.mkdir(parents=True, exist_ok=True)
    print(f"training {a.tag}: {o}", flush=True)
    st.train(cfg)


if __name__ == "__main__":
    main()
