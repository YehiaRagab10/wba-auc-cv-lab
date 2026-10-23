#!/usr/bin/env python3
"""Predict speaking vs listening per clip (test meta leaves `mode` empty).

Features: per-clip std over frames of the similarity-normalised landmarks (212) plus
the std of frame-to-frame velocity (212) -- speaking moves the mouth far more.
Logistic regression fitted on the training split only, accuracy reported on the
validation split (disjoint drivers and avatars), predictions saved for both test gens.

    .venv/Scripts/python.exe scripts/mode_classifier.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from avatar_shape_features import normalise

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/fusion/mode_pred"


def feats(split: str) -> np.ndarray:
    p = OUT / f"{split}_feats.npy"
    if p.is_file():
        return np.load(p)
    lm = np.load(ROOT / f"cache/{split}_lm.f16.npy", mmap_mode="r")
    off = np.load(ROOT / f"cache/{split}_offsets.i64.npy")
    X = np.empty((len(off) - 1, 424), np.float32)
    for i in range(len(off) - 1):
        f = normalise(np.asarray(lm[off[i]:off[i + 1]], np.float32)).reshape(-1, 212)
        X[i, :212] = f.std(0)
        X[i, 212:] = np.diff(f, axis=0).std(0) if len(f) > 1 else 0
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(p, X)
    return X


def main() -> None:
    meta = pd.read_parquet(ROOT / "cache/train_meta.parquet")
    X = feats("train")
    sp = ROOT / "data/train_val_splits"
    ld = lambda f: set(json.loads((sp / f).read_text()))  # noqa: E731
    tr = (meta["driver_id"].isin(ld("driver_train_splits.json"))
          & meta["source_id"].isin(ld("source_train_splits.json"))).to_numpy()
    va = (meta["driver_id"].isin(ld("driver_val_splits.json"))
          & meta["source_id"].isin(ld("source_val_splits.json"))).to_numpy()
    y = (meta["mode"] == "speaking").to_numpy().astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(X[tr], y[tr])
    acc = (clf.predict(X[va]) == y[va]).mean()
    print(f"train acc {(clf.predict(X[tr]) == y[tr]).mean():.4f}   val acc {acc:.4f}  ({va.sum()} clips)")
    rel = meta["relpath"].str[: -len("/landmark_2d_106.bl2")]
    pd.DataFrame({"key": rel[va], "pred": clf.predict(X[va]), "true": y[va]}).to_csv(
        OUT / "val_pred.csv", index=False)
    for g in ("test-gen1", "test-gen2"):
        p = clf.predict_proba(feats(g))[:, 1]
        np.save(OUT / f"{g}_pspeak.npy", p)
        print(f"{g}: predicted speaking share {(p > 0.5).mean():.3f}  "
              f"confident (p<0.05 or >0.95) {((p < .05) | (p > .95)).mean():.3f}")


if __name__ == "__main__":
    main()
