#!/usr/bin/env python3
"""Per-clip static face-shape descriptor, for grouping clips by avatar without labels.

Each frame's 106 landmarks are centred, scaled to unit RMS radius and rotated so the
eye line is horizontal; the clip descriptor is the per-coordinate median over frames
(the resting face, dominated by the avatar's anatomy, not the teleoperator's motion).

    .venv/Scripts/python.exe scripts/avatar_shape_features.py --split train
    .venv/Scripts/python.exe scripts/avatar_shape_features.py --split test-gen1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "output/fusion/avatar_shape"
# 106-point layout: 35-42 and 89-96 are the two eye contours in the InsightFace scheme
EYE_L, EYE_R = list(range(35, 43)), list(range(89, 97))


def normalise(f: np.ndarray) -> np.ndarray:
    """f: (T, 106, 2) -> similarity-normalised (T, 106, 2)."""
    f = f - f.mean(1, keepdims=True)
    f = f / np.sqrt((f ** 2).sum(2).mean(1))[:, None, None].clip(1e-6)
    d = f[:, EYE_R].mean(1) - f[:, EYE_L].mean(1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    c, s = np.cos(-ang)[:, None], np.sin(-ang)[:, None]
    x, y = f[..., 0], f[..., 1]
    return np.stack((c * x - s * y, s * x + c * y), -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    lm = np.load(CACHE / f"{a.split}_lm.f16.npy", mmap_mode="r")
    off = np.load(CACHE / f"{a.split}_offsets.i64.npy")
    feats = np.empty((len(off) - 1, 212), np.float32)
    for i in range(len(off) - 1):
        f = normalise(np.asarray(lm[off[i]:off[i + 1]], np.float32))
        feats[i] = np.median(f, 0).reshape(-1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{a.split}.npy", feats)
    print(f"{a.split}: {feats.shape} -> {OUT / f'{a.split}.npy'}")


if __name__ == "__main__":
    main()
