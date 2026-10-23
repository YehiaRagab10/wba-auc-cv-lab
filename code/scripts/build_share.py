#!/usr/bin/env python3
"""Build a correlation-share variant of the c65 submission.

The c65/c70 builds were made ad hoc and no recipe was written down, so the branch
set and weights here were recovered from the shipped file itself with
``scripts/identify_submission.py`` (every block matches a known branch at cosine
1.000000 and the weights sum to 1.0000).

Rescaling keeps the within-family proportions and moves only the split between the
correlation family and the rest, which is the lever the leaderboard measured:

    corr weights   *= target / current_corr_share
    other weights  *= (1 - target) / (1 - current_corr_share)

    python scripts/build_share.py --share 0.80 --tag c80
    python scripts/build_share.py --share 0.65 --verify-only   # reproduce c65

``--verify-only`` rebuilds c65 and checks it against the shipped file at cosine 1,
which is the guard that this reconstruction really is the submitted system.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import blosc2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VID = "../wba-video-best-0.8978/embeddings"

# name -> (gen1 path, weight in the c65 build, is correlation family)
C65 = {
    "c_classical":   ("output/landmarks/dcs/classical/gen1.bl2",            0.0766, True),
    "c_d8_hand":     ("output/landmarks/dcs/d8_hand_seed1/gen1.bl2",        0.0766, True),
    "c_rank":        ("output/landmarks/dcs/classical_rank/gen1.bl2",       0.0766, True),
    "c_delta":       ("output/landmarks/dcs/classical_delta/gen1.bl2",      0.0766, True),
    "c_accel":       ("output/landmarks/dcs/classical_accel/gen1.bl2",      0.0766, True),
    "c_partial":     ("output/landmarks/dcs/classical_partial/gen1.bl2",    0.0766, True),
    "c_lagged":      ("output/landmarks/dcs/classical_lagged/gen1.bl2",     0.0766, True),
    "c_band_low":    ("output/landmarks/dcs/classical_band_low/gen1.bl2",   0.0766, True),
    "c_band_high":   ("output/landmarks/dcs/classical_band_high/gen1.bl2",  0.0383, True),
    "b3":            (f"{VID}/b3_batch128_seed1-gen1-f100.bl2",             0.0655, False),
    "al1":           (f"{VID}/al1_aligned_s0-gen1-f100.bl2",                0.0655, False),
    "b5":            (f"{VID}/b5_batch128_seed2-gen1-f100.bl2",             0.0191, False),
    "f2":            (f"{VID}/f2_featdiff_b128_6000it-gen1-f100.bl2",       0.0191, False),
    "stgcn":         ("output/fusion/parity_fixed/stgcn/embeddings-gen1.bl2", 0.0655, False),
    "ktdnn_1":       ("output/video/submission_build/ktdnn_1-gen1.bl2",     0.0327, False),
    "ktdnn_2":       ("output/video/submission_build/ktdnn_2-gen1.bl2",     0.0327, False),
    "nvidia":        ("output/fusion/parity_fixed/nvidia/embeddings-gen1.bl2", 0.0491, False),
}

# Correlation members not in the c65 build. c65 -> c80 (pure reweighting) LOST
# 0.010 diff-gen, while c51 -> c65 (reweighting AND five new branches) GAINED 0.038,
# so added diversity, not added share, is what moved the leaderboard. These are the
# remaining unused correlation statistics.
EXTRA = {
    "c_band_mid":  ("output/landmarks/dcs/classical_band_mid/gen1.bl2",  0.0766, True),
    "c_coherence": ("output/landmarks/dcs/classical_coherence/gen1.bl2", 0.0766, True),
    # Coherence on time-ranked signals: invariant to any monotone per-channel map
    # AND any per-channel linear filter -- the top of the invariance ladder, and the
    # most orthogonal member we have measured (mean |r| 0.093 against the whole build).
    "c_rank_coherence": ("output/landmarks/dcs/classical_rank_coherence/gen1.bl2", 0.0766, True),
    # ktdnn trained with Yehia's avatar shape augmentation. Measured +0.0006 on the
    # c65 build and r=0.739 with ktdnn_1, so it is close to a duplicate on the
    # same-generator split -- but that correlation cannot be measured across
    # generators, which is where a shape-invariance-trained model might diverge.
    "sa_shape": ("output/landmarks/ktdnn2/sa_shape_s3/gen1.bl2", 0.0327, False),
}
ROWS = 40200


def gen2(p: str) -> str:
    return (p.replace("/gen1.bl2", "/gen2.bl2")
             .replace("-gen1-f100.bl2", "-gen2-f100.bl2")
             .replace("-gen1.bl2", "-gen2.bl2")
             .replace("embeddings-gen1.bl2", "embeddings-gen2.bl2"))


def rescale(share: float) -> dict[str, float]:
    cur = sum(w for _, w, c in C65.values() if c)
    tot = sum(w for _, w, _ in C65.values())
    a, b = share / cur, (1 - share) / (tot - cur)
    return {n: w * (a if c else b) for n, (_, w, c) in C65.items()}


def build(weights: dict[str, float], which: str) -> torch.Tensor:
    W = sum(weights.values())
    parts = []
    for n, (p, _, _) in C65.items():
        path = ROOT / (p if which == "gen1" else gen2(p))
        t = torch.as_tensor(blosc2.load_tensor(str(path))).float()
        if t.shape[0] != ROWS:
            raise SystemExit(f"{n} {which}: {t.shape[0]} rows, expected {ROWS}")
        parts.append(F.normalize(t, dim=-1) * (weights[n] / W) ** 0.5)
    out = torch.cat(parts, dim=-1)
    if not torch.isfinite(out).all() or float(out.norm(dim=-1).min()) < 0.999:
        raise SystemExit(f"{which}: non-finite or zero-norm row")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--add", nargs="*", default=[],
                    help="extra correlation members to include, e.g. c_band_mid c_coherence")
    a = ap.parse_args()
    for n in a.add:
        if n not in EXTRA:
            raise SystemExit(f"unknown extra branch {n}; have {sorted(EXTRA)}")
        C65[n] = EXTRA[n]

    cur = sum(w for _, w, c in C65.values() if c)
    print(f"c65 correlation share {cur:.4f} -> target {a.share:.2f}")
    w = rescale(a.share)
    got = sum(v for n, v in w.items() if C65[n][2]) / sum(w.values())
    print(f"rebuilt share {got:.4f}\n")
    for n, v in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {n:<14s} {v/sum(w.values()):.4f}{'  [corr]' if C65[n][2] else ''}")

    if a.verify_only:
        ref = torch.as_tensor(blosc2.load_tensor(
            str(ROOT / "output/fusion/submission_c65/embeddings-gen1.bl2"))).float()
        mine = build(rescale(cur), "gen1")
        cs = F.cosine_similarity(F.normalize(ref, dim=-1), F.normalize(mine, dim=-1), dim=-1)
        print(f"\nreproduce c65: dims {mine.shape[1]} vs {ref.shape[1]}, "
              f"cosine min {cs.min():.6f} mean {cs.mean():.6f}")
        raise SystemExit(0 if float(cs.min()) > 0.99999 else 1)

    tag = a.tag or f"c{int(round(a.share*100))}"
    work = ROOT / f"output/fusion/submission_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    for which in ("gen1", "gen2"):
        e = build(w, which)
        blosc2.save_tensor(e.contiguous(), str(work / f"embeddings-{which}.bl2"), mode="w")
        print(f"  {which}: {tuple(e.shape)}")
    (work / "recipe.json").write_text(json.dumps(
        {"tag": tag, "corr_share": got, "derived_from": "submission_c65 (recovered)",
         "weights": {n: v / sum(w.values()) for n, v in w.items()}}, indent=1), encoding="utf-8")

    out = ROOT / f"submission/wba2026-corr-{tag}.zip"
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_submission_file.py"),
                    str(work / "embeddings-gen1.bl2"), str(work / "embeddings-gen2.bl2"),
                    "cosine", "video+landmarks", "--out", str(out)], check=True)
    print(f"\nwrote {out.name} ({out.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
