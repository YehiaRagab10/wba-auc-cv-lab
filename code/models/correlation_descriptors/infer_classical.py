"""Test-set inference for the classical Approach C variant.

The classical variant is the branch that took the largest weight (0.237) in the
0.9325 fusion, despite the worst standalone score of the twelve branches, so a
submission needs its test embeddings.

It refits the §5 back-end here rather than loading the stored
``classical_transform.npz``, deliberately: the validation fusion that measured 0.9325
fitted the transform on a seeded 20,000-clip train subset through ``backend.fit_backend``,
and the stored npz was written by an earlier code path with different keys and a
different LDA centering term. Refitting with the same seed and the same function is
what makes the test embeddings consistent with the validation number — the drift that
plan §1.6 note 4 exists to prevent.

Usage
-----
    python models/landmarks/dcs/infer_classical.py --split test-gen1 \\
        --out output/landmarks/dcs/classical/gen1.bl2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import fit_backend                                          # noqa: E402
from classical import descriptors                                        # noqa: E402
from data import LandmarkCache, train_val_indices                        # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TRAIN_CLIPS = 20000      # must match scripts/dcs_fusion_eval.py classical_val_embeddings
TRAIN_SEED = 1234


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test-gen1", "test-gen2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", type=int, default=200)
    ap.add_argument("--train-clips", type=int, default=TRAIN_CLIPS)
    ap.add_argument("--signals", default="base",
                    choices=["base", "dist", "delta", "accel", "lagged", "partial",
                             "band_low", "band_mid", "band_high", "coherence"])
    ap.add_argument("--rank", action="store_true",
                    help="Spearman (rank) correlations instead of Pearson")
    ap.add_argument("--shrinkage", type=float, default=0.1)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("fitting the back-end on train (same subset as the validation fusion) ...", flush=True)
    tr_cache = LandmarkCache("train")
    tr, _ = train_val_indices(tr_cache)
    sub = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(TRAIN_SEED))
             [:a.train_clips]]
    xt, ot = descriptors(tr_cache, sub, device, signals=a.signals, rank=a.rank)
    be = fit_backend(torch.from_numpy(xt),
                     tr_cache.meta["driver_id"].to_numpy()[ot.numpy()], dims=a.dims,
                     shrinkage=a.shrinkage)
    if be is None:
        raise SystemExit("back-end could not be fitted")
    print(f"back-end: {xt.shape[1]} -> {be.dims} dims, fitted on {len(sub)} clips", flush=True)
    del tr_cache, xt

    cache = LandmarkCache(a.split)
    t0 = time.time()
    xs, order = descriptors(cache, torch.arange(len(cache)), device, signals=a.signals, rank=a.rank)
    emb = be.apply(torch.from_numpy(xs))        # already L2-normalised by Backend.apply
    print(f"embedded {len(emb)} clips in {time.time()-t0:.0f}s, dim {emb.shape[1]}", flush=True)

    # descriptors() walks length buckets, so restore official meta order.
    restored = torch.empty_like(emb)
    restored[order] = emb

    assert len(restored) == len(cache.meta), f"{len(restored)} rows vs {len(cache.meta)} meta"
    assert torch.isfinite(restored).all(), "non-finite values in embeddings"
    assert float(restored.norm(dim=-1).min()) > 1e-6, "zero-norm rows"
    restored = F.normalize(restored, p=2, dim=-1)

    import blosc2
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blosc2.save_tensor(restored.float().contiguous(), str(dest), mode="w")
    print(f"wrote {dest} {tuple(restored.shape)} float32", flush=True)


if __name__ == "__main__":
    main()
