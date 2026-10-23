#!/usr/bin/env python3
"""Per-avatar transductive centering of one test generation, branch by branch.

TRANSDUCTIVE, like build_transductive.py: each embedding depends on other test clips
(no labels, no external data). Each branch is L2-normalised, the mean of the clip's k
nearest neighbours in static face-shape space is subtracted (an estimate of "what this
avatar's face does to every teleoperator"), the branch is re-normalised and weighted
exactly as in the uncentred fusion.

Measured on validation (scripts/avatar_centering_eval.py), c51 recipe:
    4-pair 0.9233 -> 0.9375 with k = 0.5-1x the avatar group size (oracle labels 0.9372);
    top-200 impostors sharing an avatar 42.5% -> 19-23%.
Test structure (scripts/avatar_groups_test.py): ~50 avatars x ~804 clips per generation,
so k = 402 is the validated half-group ratio.

    .venv/Scripts/python.exe scripts/build_avatar_centered.py --verify
    .venv/Scripts/python.exe scripts/build_avatar_centered.py --gen gen1 --k 402
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import blosc2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from build_share import C65, EXTRA, gen2

ROOT = Path(__file__).resolve().parents[1]
PATHS = {n: p for n, (p, _, _) in {**C65, **EXTRA}.items()}
# overnight seeds (scripts/overnight_seeds.sh), exported with the shipped recipe (f100, TTA 5)
for _n in ("al2_aligned_s1", "al3_aligned_s2", "b7_batch128_seed3", "al4_aligned_s3", "al5_aligned_s4",
           "alf1_featdiff_aligned", "ala1_attn_aligned", "alc1_crossmode_aligned"):
    PATHS[_n] = f"output/video/submission_build/{_n}-gen1-f100.bl2"
# NVIDIA TCNN variants (scripts/train_nvidia_variant.py + export_fusion_landmarks.py --checkpoint)
for _n in ("nv_t01_s1", "nv_t05_s1", "nv_t01_s2"):
    PATHS[_n] = f"output/fusion/variants/{_n}/embeddings-gen1.bl2"
# corr-accel-c51 (submission 964275, same-gen 0.888), weights from the recorded recipe
# (block order as recovered from the shipped file by identify_submission.py)
C51 = {"c_classical": 0.18462962962962962, "b3": 0.11212962962962962,
       "al1": 0.11212962962962962, "stgcn": 0.09342592592592593,
       "c_d8_hand": 0.07101851851851852, "c_delta": 0.07101851851851852,
       "c_rank": 0.05675925925925925, "c_partial": 0.05675925925925925,
       "ktdnn_1": 0.056018518518518516, "ktdnn_2": 0.056018518518518516,
       "nvidia": 0.056018518518518516, "c_accel": 0.07407407407407407}
ROWS = 40200


def shape_neighbours(gen: str, k: int) -> torch.Tensor:
    meta = pd.read_parquet(ROOT / "cache/train_meta.parquet")
    shape = np.load(ROOT / "output/fusion/avatar_shape/train.npy")
    sp = ROOT / "data/train_val_splits"
    load = lambda f: set(json.loads((sp / f).read_text()))  # noqa: E731
    tr = (meta["driver_id"].isin(load("driver_train_splits.json"))
          & meta["source_id"].isin(load("source_train_splits.json"))).to_numpy()
    lda = LinearDiscriminantAnalysis(n_components=64).fit(shape[tr], meta["source_id"][tr])
    z = torch.from_numpy(lda.transform(np.load(ROOT / f"output/fusion/avatar_shape/test-{gen}.npy"))
                         .astype(np.float32))
    z = F.normalize(z, dim=-1)
    nn = torch.empty(len(z), k, dtype=torch.long)
    for i in range(0, len(z), 2048):
        nn[i:i + 2048] = (z[i:i + 2048] @ z.T).topk(k, dim=1).indices
    return nn


def branch(name: str, gen: str) -> torch.Tensor:
    p = PATHS[name]
    t = torch.as_tensor(blosc2.load_tensor(str(ROOT / (p if gen == "gen1" else gen2(p))))).float()
    assert t.shape[0] == ROWS, f"{name} {gen}: {t.shape[0]} rows"
    return F.normalize(t, dim=-1)


C65W = {n: w for n, (_, w, _) in C65.items()}   # the 17-branch c65 build (c65-transcenter)
# c51 + b5 (second seed of b3) at 0.112: val 0.9654 -> 0.9674 with avatar + mode + DBA8x2,
# +0.0020 on held-out drivers, 6/6 splits
C51B5 = {**C51, "b5": 0.112}
# + al2 (new aligned seed) at 0.224: val 0.9674 -> 0.9725, +0.0051 held-out, 6/6 splits
C51B5AL2 = {**C51B5, "al2_aligned_s1": 0.224}
# + al3 (aligned seed 2, standalone 0.8696) at 0.112: 0.9725 -> 0.9740, +0.0015 held-out, 6/6
C51B5AL23 = {**C51B5AL2, "al3_aligned_s2": 0.112}
# + alf1 (feature differencing on aligned crops, 6000 it) at 0.224: 0.9740 -> 0.9758,
# +0.0029 held-out, 6/6 splits
C51B5AL23F = {**C51B5AL23, "alf1_featdiff_aligned": 0.224}


def build(weights: dict[str, float], gen: str, nn: torch.Tensor | None,
          corr_global: bool = False, mode: torch.Tensor | None = None) -> torch.Tensor:
    """corr_global: correlation branches get plain per-generation centering instead of
    per-avatar centering (on validation avatar centering costs them identity: corr
    family 0.804 -> 0.764, while neural branches gain, e.g. stgcn 0.794 -> 0.835)."""
    W = sum(weights.values())
    parts = []
    for n, w in weights.items():
        e = branch(n, gen)
        if corr_global and n.startswith("c_"):
            e = F.normalize(e - e.mean(0, keepdim=True), dim=-1)
        elif nn is not None:
            m = torch.empty_like(e)
            for i in range(0, ROWS, 1024):
                m[i:i + 1024] = e[nn[i:i + 1024]].mean(1)
            e = F.normalize(e - m, dim=-1)
        if mode is not None:            # subtract the mean of clips with the same (predicted) mode
            for v in (0, 1):
                sel = mode == v
                e[sel] = e[sel] - e[sel].mean(0, keepdim=True)
            e = F.normalize(e, dim=-1)
        parts.append(e * (w / W) ** 0.5)
        print(f"  {n:<12s} dim {e.shape[1]:5d}  weight {w / W:.4f}")
    out = torch.cat(parts, dim=-1)
    assert torch.isfinite(out).all() and float(out.norm(dim=-1).min()) > 0.999
    return out


def voted_mode(e: torch.Tensor, gen: str, k: int = 8) -> torch.Tensor:
    """Speaking(1)/listening(0) per clip: classifier (scripts/mode_classifier.py, val acc
    0.953) majority-voted over the clip's top-k fused neighbours, which are mostly the
    same recording on other avatars (val acc 0.979)."""
    pred = torch.from_numpy((np.load(ROOT / f"output/fusion/mode_pred/test-{gen}_pspeak.npy") > 0.5)
                            .astype(np.float32))
    out = torch.empty(len(e))
    for i in range(0, len(e), 1024):
        _, j = (e[i:i + 1024] @ e.T).topk(k, dim=1)
        out[i:i + 1024] = (pred[j].mean(1) > 0.5).float()
    return out


def dba(e: torch.Tensor, k: int, p: float = 1.0) -> torch.Tensor:
    """Database-side augmentation: each embedding becomes the cos^p-weighted mean of its
    top-k fused neighbours (self included). Validated in scripts/postproc_eval.py:
    k=8, p=1 -> 4-pair 0.9376 -> 0.9554; +0.015 on held-out drivers; gain flat
    (+0.017..+0.020) from 5 to 20 identities."""
    out = torch.empty_like(e)
    for i in range(0, len(e), 1024):
        v, j = (e[i:i + 1024] @ e.T).topk(k, dim=1)
        w = v.clamp_min(0) ** p
        out[i:i + 1024] = (w[..., None] * e[j]).sum(1)
    return F.normalize(out, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="gen1")
    ap.add_argument("--k", type=int, default=402)
    ap.add_argument("--dba-k", type=int, default=0, help="DBA neighbours (0 = off)")
    ap.add_argument("--dba-rounds", type=int, default=1,
                    help="apply DBA this many times (2: +0.001 held-out on val, 6/6 splits; "
                         "improves all three calibrated gen2 proxies)")
    ap.add_argument("--corr-share", type=float, default=0.0,
                    help="rescale the correlation-family share of the recipe (0 = as is)")
    ap.add_argument("--mode-removal", action="store_true",
                    help="also subtract per-(predicted)-mode means per branch "
                         "(val: with DBA8x2 0.9654, +0.0022 held-out, 6/6 splits)")
    ap.add_argument("--recipe", choices=["c51", "c51b5", "c51b5al2", "c51b5al23", "c51b5al23f", "c65"], default="c51")
    ap.add_argument("--corr-global", action="store_true",
                    help="global (not per-avatar) centering for correlation branches")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild c51 uncentred and compare with the submitted gen1 file")
    a = ap.parse_args()

    if a.verify:
        ref = torch.as_tensor(blosc2.load_tensor(
            str(ROOT / "output/fusion/submission_accel/embeddings-gen1.bl2"))).float()
        mine = build(C51, "gen1", None)
        cs = F.cosine_similarity(F.normalize(ref, dim=-1), mine, dim=-1)
        print(f"reproduce c51: dims {mine.shape[1]} vs {ref.shape[1]}, cosine min {cs.min():.6f}")
        raise SystemExit(0 if float(cs.min()) > 0.99999 else 1)

    nn = shape_neighbours(a.gen, a.k)
    weights = dict({"c51": C51, "c51b5": C51B5, "c51b5al2": C51B5AL2, "c51b5al23": C51B5AL23, "c51b5al23f": C51B5AL23F, "c65": C65W}[a.recipe])
    if a.corr_share:
        c = sum(v for n, v in weights.items() if n.startswith("c_"))
        t = sum(weights.values())
        weights = {n: v * (a.corr_share / c * t if n.startswith("c_") else (1 - a.corr_share) / (t - c) * t)
                   for n, v in weights.items()}
    e = build(weights, a.gen, nn, a.corr_global)
    if a.mode_removal:
        mode = voted_mode(e, a.gen)
        print(f"voted mode: speaking share {mode.mean():.3f}")
        e = build(weights, a.gen, nn, a.corr_global, mode)
    tag = (f"submission_{a.recipe}" + (f"_s{int(round(a.corr_share * 100))}" if a.corr_share else "")
           + f"_avatarknn{a.k}" + ("_corrglobal" if a.corr_global else "")
           + ("_moderm" if a.mode_removal else ""))
    if a.dba_k > 1:
        for _ in range(a.dba_rounds):
            e = dba(e, a.dba_k)
        tag += f"_dba{a.dba_k}" + (f"x{a.dba_rounds}" if a.dba_rounds > 1 else "")
    out = ROOT / f"output/fusion/{tag}"
    out.mkdir(parents=True, exist_ok=True)
    blosc2.save_tensor(e.contiguous(), str(out / f"embeddings-{a.gen}.bl2"), mode="w")
    (out / "recipe.json").write_text(json.dumps(
        {"base": a.recipe, "transductive": "per-avatar kNN centering",
         "corr_global": a.corr_global,
         "k": a.k, "dba_k": a.dba_k, "dba_rounds": a.dba_rounds, "mode_removal": a.mode_removal, "corr_share": a.corr_share, "shape_space": "LDA(64) on training avatars, static median shape",
         "weights": weights}, indent=1), encoding="utf-8")
    print(f"{a.gen}: {tuple(e.shape)} -> {out}")


if __name__ == "__main__":
    main()
