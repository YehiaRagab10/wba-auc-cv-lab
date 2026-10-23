"""Classical baseline inside Approach C (plan §4) — no GPU training, a few minutes.

    handcrafted 1128 correlations + 48 log-stds
      -> standardise (train statistics)
      -> LDA to min(#drivers - 1, 200) dims
      -> WCCN whitening
      -> cosine

Two jobs. It is a sanity floor: if the learned DCS net cannot beat this, the net is
broken rather than the idea. And it has a very different error pattern from a
trained network, which makes it a cheap fusion partner.

Everything is fitted on the training split only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import fit_backend                                         # noqa: E402
from data import LandmarkCache, train_val_indices                      # noqa: E402
from evaluate import mean4, official_score, pair_metrics, source_probe  # noqa: E402
from geometry import GeometricFrontEnd                                  # noqa: E402
from model import CorrelationSignature                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/landmarks/dcs"


@torch.no_grad()
def descriptors(cache: LandmarkCache, seq_idx: torch.Tensor, device: torch.device,
                batch: int = 256, signals: str = "base",
                rank: bool = False) -> tuple[np.ndarray, torch.Tensor]:
    """Handcrafted correlation + log-std descriptor per sequence.

    ``signals="base"`` uses the 48 channels of plan §4.1 (6 pose + 42 region), giving
    1128 correlations + 48 log-stds = 1176 dims. ``signals="dist"`` adds the 21
    inter-region distances, 69 signals -> 2346 + 69 = 2415 dims. The distances are
    handcrafted and PCA-free, so including them stays within the no-PCA constraint;
    they were left out originally only because §4.1 specified 48.
    """
    from geometry import SLICE_DIST
    fe = GeometricFrontEnd().to(device)
    sig = CorrelationSignature()
    # Spearman = Pearson on the time-ranks. Pearson is invariant to per-channel affine
    # change (the gain-invariance Approach C rests on); ranks are invariant to ANY
    # monotone transform, so a generator that distorts motion non-linearly breaks
    # Pearson and leaves Spearman untouched. Strictly stronger invariance, same pipeline.
    def _rank_time(t):
        order = t.argsort(dim=-1).argsort(dim=-1).float()
        n = t.shape[-1]
        return (order - (n - 1) / 2) / max(1.0, (n - 1) / 2)
    idx = fe.handcrafted_channels().to(device)
    if signals == "dist":
        idx = torch.cat([idx, torch.arange(SLICE_DIST.start, SLICE_DIST.stop,
                                           device=device)])
    elif signals == "accel":
        # Correlations of ACCELERATION. c_delta (velocities) turned out to be the single
        # best correlation member, so the second derivative is the obvious sibling: the
        # front-end stacks [base, delta, delta-delta], so this is two N_BASE offsets.
        from geometry import N_BASE as _NB
        idx = idx + 2 * _NB
    elif signals == "delta":
        # The same 48 signals, but their first derivative. Correlations between
        # VELOCITIES describe coordination of motion rather than of posture - a
        # different physical quantity with the same gain-invariance. The front-end
        # stacks [base, delta, delta-delta], so the delta block is one N_BASE offset.
        from geometry import N_BASE
        idx = idx + N_BASE
    out, order = [], []
    t0 = time.time()
    buckets = cache.length_buckets(seq_idx, batch)
    for i, b in enumerate(buckets):
        s = fe(cache.gather_full(b, device))[:, idx]
        if signals.startswith("band"):
            # Correlations WITHIN one temporal band. Coordination at 0-1 Hz (head sway)
            # is a different phenomenon from 3-8 Hz (blinks, micro-motion), so band-
            # limited correlations should carry decorrelated identity information.
            lo, hi = {"band_low": (0.0, 1.0), "band_mid": (1.0, 3.0),
                      "band_high": (3.0, 8.0)}[signals]
            T = s.shape[-1]
            freq = torch.fft.rfftfreq(T, d=1.0 / 30.0, device=s.device)   # 30 fps
            keep = ((freq >= lo) & (freq < hi)).to(s.dtype)
            s = torch.fft.irfft(torch.fft.rfft(s, dim=-1) * keep, n=T, dim=-1)
        if rank:
            s = _rank_time(s)
        if signals == "coherence":
            # Magnitude-squared coherence, averaged over Welch segments.
            #   C_xy(f) = |S_xy(f)|^2 / (S_xx(f) S_yy(f))
            # Pearson correlation is invariant to per-channel GAIN. Coherence is
            # invariant to any per-channel LINEAR FILTER - gain, smoothing and lag
            # together - because the filter's transfer function cancels in the ratio.
            # If a different generator filters channels differently, which is exactly
            # what temporal smoothing and transfer lag look like, coherence survives
            # where Pearson does not. Strictly stronger invariance than anything else here.
            T = s.shape[-1]
            nseg, seg = 4, T // 4
            segs = s[:, :, :nseg * seg].reshape(s.shape[0], s.shape[1], nseg, seg)
            segs = segs - segs.mean(dim=-1, keepdim=True)
            win = torch.hann_window(seg, device=s.device, dtype=s.dtype)
            F = torch.fft.rfft(segs * win, dim=-1)                    # [B,M,nseg,f]
            # cross-spectrum averaged over segments, then collapsed over frequency
            Fp = F.permute(0, 2, 1, 3)                                # [B,nseg,M,f]
            cross = torch.einsum('bsmf,bsnf->bmnf', Fp, Fp.conj()) / nseg
            auto = cross.diagonal(dim1=1, dim2=2).permute(0, 2, 1)    # [B,M,f]
            denom = (auto[:, :, None, :] * auto[:, None, :, :]).real.clamp_min(1e-12)
            coh = (cross.abs() ** 2) / denom                          # [B,M,M,f]
            coh = coh.mean(dim=-1).clamp(0.0, 1.0)                    # average over frequency
            from model import _upper_tri, _fisher_z
            d = torch.cat([_fisher_z(_upper_tri(coh.real)), sig.log_std(s)], dim=-1)
        elif signals == "partial":
            # Partial correlations: invert the (ridge-regularised) covariance, so each
            # pair is conditioned on all other signals. Ordinary correlation is inflated
            # by whatever drives everything at once - global head motion, speech rate -
            # and partial correlation strips that out, leaving the DIRECT couplings.
            # A different statistic from the same signals, so decorrelated by construction.
            x = s - s.mean(dim=-1, keepdim=True)
            cov = (x @ x.transpose(1, 2)) / x.shape[-1]
            d_ = cov.diagonal(dim1=1, dim2=2).clamp_min(1e-10).sqrt()
            cov = cov / (d_[:, :, None] * d_[:, None, :])
            m_ = cov.shape[-1]
            eye = torch.eye(m_, device=cov.device).expand_as(cov)
            prec = torch.linalg.inv(cov + 0.1 * eye)          # ridge keeps it invertible
            pd = prec.diagonal(dim1=1, dim2=2).clamp_min(1e-10).sqrt()
            pc = -prec / (pd[:, :, None] * pd[:, None, :])
            from model import _upper_tri, _fisher_z
            d = torch.cat([_fisher_z(_upper_tri(pc)), sig.log_std(s)], dim=-1)
        elif signals == "lagged":
            # Lead/lag structure only: which signal moves BEFORE which, with the
            # instantaneous correlations dropped entirely. Timing is a different
            # identity cue from co-variation and is what §4 lags were meant to capture.
            d = torch.cat([sig.lagged(s), sig.log_std(s)], dim=-1)
        else:
            d = torch.cat([sig.correlation(s), sig.log_std(s)], dim=-1)
        out.append(torch.nan_to_num(d).float().cpu())
        order.append(b)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(buckets)} buckets  {time.time()-t0:.0f}s", flush=True)
    return torch.cat(out).numpy(), torch.cat(order)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=0, help="0 = use the whole train split")
    ap.add_argument("--dims", type=int, default=200)
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--signals", choices=["base", "dist", "delta", "accel", "lagged", "partial",
                             "band_low", "band_mid", "band_high", "coherence"], default="base")
    ap.add_argument("--rank", action="store_true",
                    help="Spearman (rank) correlations instead of Pearson")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = LandmarkCache("train")
    tr, va = train_val_indices(cache)
    if a.train_limit:
        tr = tr[torch.randperm(len(tr))[:a.train_limit]]
    print(f"train {len(tr)} clips, val {len(va)} clips", flush=True)

    print("computing val descriptors ...", flush=True)
    xv, ov = descriptors(cache, va, device, signals=a.signals, rank=a.rank)
    print("computing train descriptors ...", flush=True)
    xt, ot = descriptors(cache, tr, device, signals=a.signals, rank=a.rank)
    print(f"descriptor dim {xt.shape[1]}", flush=True)

    labels = cache.meta["driver_id"].to_numpy()[ot.numpy()]
    t0 = time.time()
    t = fit_backend(torch.from_numpy(xt), labels, dims=a.dims, shrinkage=a.shrinkage)
    print(f"LDA+WCCN fitted in {time.time()-t0:.0f}s -> {t.dims} dims", flush=True)

    rows_v = cache.meta.iloc[ov.numpy()]
    results = {}
    for name, emb in (("raw-cosine", torch.from_numpy(xv).float()),
                      ("lda-wccn", t.apply(torch.from_numpy(xv)))):
        m = pair_metrics(emb, rows_v)
        results[name] = m
        print(f"  {name:12s} 3-cond {m['score3']:.4f} | 4-pair {m['score4']:.4f} | "
              f"src-probe {source_probe(emb, rows_v):.3f}", flush=True)
        for k in sorted(k for k in m if k.endswith('_auc')):
            print(f"      {k.replace('emb_','').replace('_auc',''):26s} {m[k]:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classical_baseline.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, indent=1), encoding="utf-8")
    # Stored as plain arrays so applying it later needs no sklearn.
    np.savez(OUT / "classical_transform.npz", **t.state())
    print(f"wrote {OUT/'classical_baseline.json'}", flush=True)


if __name__ == "__main__":
    main()
