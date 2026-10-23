"""§1.2 GPU batch assembly and §1.5 sampler, on top of the §1.1 cache.

No DataLoader, no worker processes: the whole split lives in one flat fp16 tensor
in RAM, a batch is one fancy-index into it, and every transform afterwards runs on
the GPU. A 256-clip batch is assembled by index arithmetic alone.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache"
VERTICES = 106


def _reflect(idx: Tensor, length: Tensor) -> Tensor:
    """Reflect out-of-range frame indices back inside [0, length-1].

    Pure index arithmetic, so short sequences cost no copies. Period 2L-2 is the
    standard reflection period that does not duplicate the endpoints.
    """
    period = (2 * length - 2).clamp_min(1)
    idx = idx % period
    return torch.where(idx >= length, period - idx, idx)


class LandmarkCache:
    """One cached split, held as a flat ``[N_frames, 106, 2]`` fp16 CPU tensor."""

    def __init__(self, split: str, verbose: bool = True) -> None:
        self.split = split
        lm = np.load(CACHE / f"{split}_lm.f16.npy", mmap_mode="r")
        self.lm = torch.from_numpy(np.array(lm))             # real copy: random access must not page from disk
        self.offsets = torch.from_numpy(np.load(CACHE / f"{split}_offsets.i64.npy"))
        self.meta = pd.read_parquet(CACHE / f"{split}_meta.parquet")
        self.lengths = torch.as_tensor(self.meta["length"].to_numpy(), dtype=torch.long)
        if verbose:
            print(f"[cache] {split}: {len(self.meta)} seqs, {self.lm.shape[0]} frames, "
                  f"{self.lm.numel() * 2 / 2**30:.2f} GiB", flush=True)

    def __len__(self) -> int:
        return len(self.meta)

    def gather(self, seq_idx: Tensor, frames: int, device: torch.device,
               random_crop: bool = True) -> Tensor:
        """Assemble ``[B, frames, 106, 2]`` for the given sequences.

        ``random_crop`` picks a uniform start per sequence (training); otherwise the
        crop is centred (deterministic evaluation). Sequences shorter than ``frames``
        are reflect-padded.
        """
        lens = self.lengths[seq_idx]
        span = (lens - frames + 1).clamp_min(1)
        if random_crop:
            starts = (torch.rand(len(seq_idx)) * span).long()
        else:
            starts = ((lens - frames) // 2).clamp_min(0)
        local = _reflect(starts[:, None] + torch.arange(frames), lens[:, None])
        flat = (self.offsets[seq_idx][:, None] + local).reshape(-1)
        out = self.lm[flat].view(len(seq_idx), frames, VERTICES, 2)
        return out.to(device, non_blocking=True).float()

    def gather_full(self, seq_idx: Tensor, device: torch.device) -> Tensor:
        """Assemble whole sequences that all share one length (§1.6 bucketing)."""
        lens = self.lengths[seq_idx]
        n = int(lens[0])
        assert bool((lens == n).all()), "gather_full requires one exact-length bucket"
        flat = (self.offsets[seq_idx][:, None] + torch.arange(n)).reshape(-1)
        return self.lm[flat].view(len(seq_idx), n, VERTICES, 2).to(device, non_blocking=True).float()

    def length_buckets(self, seq_idx: Tensor, max_batch: int) -> list[Tensor]:
        """Group sequences into batches of identical length, at most ``max_batch`` each."""
        lens = self.lengths[seq_idx]
        out: list[Tensor] = []
        for n in lens.unique().tolist():
            sel = seq_idx[lens == n]
            out += [sel[i:i + max_batch] for i in range(0, len(sel), max_batch)]
        return out


def train_val_indices(cache: LandmarkCache) -> tuple[Tensor, Tensor]:
    """Strict split: train excludes every held-out driver *and* every held-out source."""
    dv = set(json.loads((ROOT / "data/train_val_splits/driver_val_splits.json").read_text()))
    sv = set(json.loads((ROOT / "data/train_val_splits/source_val_splits.json").read_text()))
    d = cache.meta["driver_id"].to_numpy()
    s = cache.meta["source_id"].to_numpy()
    in_dv = np.isin(d, list(dv))
    in_sv = np.isin(s, list(sv))
    train = np.flatnonzero(~in_dv & ~in_sv)
    val = np.flatnonzero(in_dv & in_sv)
    return torch.from_numpy(train), torch.from_numpy(val)


class TrialSampler:
    """§1.5 sampler — batches shaped like the test trials.

    Impostor pairs in the test set share the target appearance, so a batch is built
    source-first: pick ``sources`` targets, pick drivers that appear with those
    targets, and take several clips per driver spanning at least two sources and both
    modes. Every anchor therefore has positives across targets and modes, and hard
    negatives that share its target. Self-reenactments are kept (they exist in the
    gallery/probe pools, so training should see them).
    """

    def __init__(self, cache: LandmarkCache, indices: Tensor, drivers: int = 32,
                 clips_per_driver: int = 8, sources: int = 8, seed: int = 0) -> None:
        self.cache, self.n_drivers = cache, drivers
        self.clips_per_driver, self.n_sources = clips_per_driver, sources
        self.rng = random.Random(seed)

        meta = cache.meta.iloc[indices.numpy()]
        self.driver_ids = sorted(meta["driver_id"].unique())
        self.source_ids = sorted(meta["source_id"].unique())
        self.driver_label = {d: i for i, d in enumerate(self.driver_ids)}
        self.source_label = {s: i for i, s in enumerate(self.source_ids)}

        # (driver, source, mode) -> seq indices; plus the lookups the sampler walks.
        self.groups: dict[tuple[str, str, str], list[int]] = {}
        self.by_source: dict[str, set[str]] = {}
        self.by_driver: dict[str, set[str]] = {}
        for seq, drv, src, mode in zip(meta["seq_idx"], meta["driver_id"],
                                       meta["source_id"], meta["mode"]):
            self.groups.setdefault((drv, src, mode), []).append(int(seq))
            self.by_source.setdefault(src, set()).add(drv)
            self.by_driver.setdefault(drv, set()).add(src)

    @property
    def num_drivers(self) -> int:
        return len(self.driver_ids)

    @property
    def num_sources(self) -> int:
        return len(self.source_ids)

    def _pick_drivers(self) -> tuple[list[str], set[str]]:
        """Choose drivers target-first, so several drivers share each target.

        Returns the drivers and the anchor targets they were drawn from.
        """
        per_source = max(1, self.n_drivers // self.n_sources)
        chosen: list[str] = []
        anchors = set(self.rng.sample(self.source_ids, min(self.n_sources, len(self.source_ids))))
        for src in anchors:
            cand = [d for d in self.by_source[src] if d not in chosen]
            self.rng.shuffle(cand)
            chosen += cand[:per_source]
        pool = [d for d in self.driver_ids if d not in chosen]
        self.rng.shuffle(pool)
        chosen += pool[:max(0, self.n_drivers - len(chosen))]
        return chosen[:self.n_drivers], anchors

    def _pick_clips(self, driver: str, anchors: set[str]) -> list[int]:
        """Clips for one driver, alternating mode on every pick and rotating sources.

        The alternation is explicit rather than a round-robin over a sorted key list:
        with the keys sorted by mode, the first ``clips_per_driver`` of them are all
        the same mode, which silently starves the cross-mode pairs.

        Within a mode, this driver clips on the batch anchor targets are offered
        first, so the impostor pairs stay concentrated on shared appearances instead
        of being diluted across every target the driver happens to have.
        """
        by_mode: dict[str, list[tuple[str, str, str]]] = {}
        for k in self.groups:
            if k[0] == driver:
                by_mode.setdefault(k[2], []).append(k)
        if not by_mode:
            return []
        modes = sorted(by_mode)
        for v in by_mode.values():
            self.rng.shuffle(v)
            v.sort(key=lambda k: k[1] not in anchors)      # anchor targets first

        picks: list[int] = []
        i = 0
        while len(picks) < self.clips_per_driver:
            group = by_mode[modes[i % len(modes)]]
            key = group[(i // len(modes)) % len(group)]
            picks.append(self.rng.choice(self.groups[key]))
            i += 1
        return picks

    def batch(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return (seq_idx, driver_label, source_label) for one batch."""
        seqs, dlab, slab = [], [], []
        meta = self.cache.meta
        drivers, anchors = self._pick_drivers()
        for drv in drivers:
            for seq in self._pick_clips(drv, anchors):
                seqs.append(seq)
                dlab.append(self.driver_label[drv])
                slab.append(self.source_label[meta["source_id"].iloc[seq]])
        return (torch.tensor(seqs), torch.tensor(dlab), torch.tensor(slab))
