#!/usr/bin/env python3
"""Motion-differencing video encoder for WBA2026 (video branch).

Each clip is a stack of grayscale face crops (see preprocess.py). The network embeds *how the
face moves* while cancelling *what the avatar looks like* by subtracting consecutive frames:

    variant "pixdiff":  encode(x[t+1] - x[t])            (difference in pixel space)
    variant "featdiff": encode(x[t+1]) - encode(x[t])    (difference in feature space)

Design follows the published idea of inter-frame feature differencing (arXiv 2604.23247); this
is an independent implementation, trained from scratch on the competition training data only.

Examples (project root, venv active, MLFLOW_TRACKING_URI=sqlite:///mlflow.db on Windows):
    python models/video/mdnet/trainer.py --variant pixdiff
    python models/video/mdnet/trainer.py --variant featdiff --iterations 30000
    python models/video/mdnet/trainer.py --test --checkpoint output/video/mdnet_featdiff/best_model.pt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))  # models/  (tracking.py)
sys.path.insert(0, str(SCRIPT_DIR))                # this folder (preprocess.py)

from preprocess import load_crop  # noqa: E402
from tracking import CSVMetricsLogger, setup_mlflow_run  # noqa: E402
from extensions import RegionalTemporalBranch, cross_mode_positive_loss  # noqa: E402

import mlflow  # noqa: E402

LOGGER = logging.getLogger("mdnet")
SPLIT_DIR = PROJECT_ROOT / "data/train_val_splits"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    cache_root: Path = PROJECT_ROOT / "data/video_crops/train_64"
    meta_path: Path = PROJECT_ROOT / "data/meta/train-mp4-meta.json"
    train_driver_split: Path = SPLIT_DIR / "driver_train_splits.json"
    train_source_split: Path = SPLIT_DIR / "source_train_splits.json"
    val_driver_split: Path = SPLIT_DIR / "driver_val_splits.json"
    val_source_split: Path = SPLIT_DIR / "source_val_splits.json"
    output_dir: Path = PROJECT_ROOT / "output/video/mdnet_featdiff"
    variant: str = "featdiff"        # "pixdiff" | "featdiff"
    use_mix: bool = True             # global row/column mixing block
    use_graph: bool = True           # channel-space kNN graph block
    image_size: int = 64
    frames: int = 64                 # training window (frames at 30 fps)
    embedding_dim: int = 256
    batch_drivers: int = 8
    batch_sources: int = 4
    cross_samples: int = 2           # batch = drivers * sources * cross_samples = 64 clips
    iterations: int = 30_000
    validation_interval: int = 1_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 500
    min_lr: float = 1e-6
    temperature: float = 0.07
    ema_decay: float = 0.999         # 0 = off
    patience: int = 10               # early stopping, in validation intervals (0 = off)
    window_stride: int = 36          # sliding-window stride for evaluation
    workers: int = 6
    seed: int = 0
    amp: bool = True
    eval_distance: str = "cosine"
    register: bool = True            # at the end of training: upload the best model to the MLflow Model Registry
    # --- generalisation options (all use the training data only) ---
    input_size: int = 0              # network input resolution; 0 = image_size. If smaller than the cached crops
                                     # (e.g. cache 96 -> input 64) crops are resized, and augmentation zooms/shifts them
    augment: bool = False            # renderer-style augmentation: photometric, blur, noise, affine, speed, motion amplitude
    speed_min: float = 0.8           # temporal speed change range (needs augment)
    speed_max: float = 1.25
    mixstyle: float = 0.0            # probability of MixStyle (per-clip feature statistics mixing) in a training step
    embed_bn: bool = False           # BatchNorm on the embedding (statistics learned on training data, frozen at test)
    uniformity: float = 0.0          # weight of a loss that spreads embeddings of different drivers over the sphere
    run_tag: str = ""                # appended to the MLflow run name (e.g. for ablations)
    group: str = ""                  # ablation-study group (MLflow tag + study log)
    aug_parts: str = "amp,photo,blur,spatial,noise,speed"   # active augmentations (needs augment)
    aug_strength: float = 1.0        # scales every augmentation towards the identity
    diff_lags: str = "1"             # pixdiff: temporal difference lags, e.g. "1,2,4" (multi-scale motion)
    temporal_pool: str = "avg"       # "avg" | "attn": how the 3D head pools over time
    shift_eval: bool = True          # at the end of training: robustness proxy on synthetic renderer shifts
    width: float = 1.0               # channel multiplier of the whole network (capacity study)
    grad_checkpoint: bool = False    # recompute activations in the backward pass (less GPU memory, ~30% slower)
    aux_landmark: float = 0.0        # weight of the auxiliary "predict the landmark embedding" loss (0 = off)
    aux_landmark_path: Path = PROJECT_ROOT / "output/video/aux/stgcn_train_embeddings.pt"
    deflicker: bool = False          # normalise each frame to the clip's mean/std before differencing
    consistency: float = 0.0         # weight of the clean-vs-perturbed embedding agreement term (0 = off)
    consistency_parts: str = "photo,blur,noise"   # rendering-only perturbations; never amp/speed/spatial
    consistency_strength: float = 0.5
    arch: str = "2d"                # "2d" (per-frame conv + late 3D head, default) or "3d" (early spatio-temporal convs)
    grad_cache: int = 0              # >0: GradCache chunk size (clips); batch size no longer limited by GPU memory
    mode_balance: bool = False       # every driver-source pair contributes speaking AND listening clips (cross-mode positives)
    cross_mode_weight: float = 0.0   # mix base SupCon with cross-mode positives, preserving loss scale
    temporal_branch: bool = False    # four-region dilated temporal residual branch (2D only)
    schedule_iterations: int = 0     # LR horizon for truncated screens; 0 = iterations
    io_threads: int = 0              # shared-process crop decoding; random cropping remains ordered


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """Cached face crops filtered by driver/source split (and by cache availability)."""

    def __init__(self, cfg: Settings, driver_split: Path, source_split: Path, train: bool) -> None:
        drivers = set(json.loads(driver_split.read_text()))
        sources = set(json.loads(source_split.read_text()))
        self.root, self.frames, self.train = cfg.cache_root, cfg.frames, train
        self.io_threads = cfg.io_threads if train else 0
        self._io_pool = None
        self.augment = cfg.augment and train and "speed" in cfg.aug_parts.split(",")
        self.speed = (cfg.speed_min, cfg.speed_max)
        self.items = [
            dict(m) for m in json.loads(cfg.meta_path.read_text())
            if m["driver_id"] in drivers and m["source_id"] in sources
            and (self.root / (m["relpath"][:-4] + ".b2c")).is_file()
        ]
        ids: dict[str, int] = {}
        for prefix in ("driver", "source"):
            for it in self.items:
                key = f"{prefix}:{it[prefix + '_id']}"
                ids.setdefault(key, len(ids))
                it[prefix + "_idx"] = ids[key]
        LOGGER.info("%s set: %d clips, %d drivers, %d sources",
                    "train" if train else "val", len(self.items),
                    len({i["driver_id"] for i in self.items}), len({i["source_id"] for i in self.items}))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._make_item(index, self._load(index))

    def _load(self, index: int) -> np.ndarray:
        return load_crop(self.root / (self.items[index]["relpath"][:-4] + ".b2c"))

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if self.io_threads <= 0:
            return [self[i] for i in indices]
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(max_workers=self.io_threads)
        # map preserves ordering; all random draws happen here, on the calling thread.
        return [self._make_item(i, clip) for i, clip in zip(indices, self._io_pool.map(self._load, indices))]

    def close(self) -> None:
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=True)
            self._io_pool = None

    def _make_item(self, index: int, clip: np.ndarray) -> dict[str, Any]:
        it = self.items[index]
        if self.train:
            t = clip.shape[0]
            if self.augment:  # speed change: resample `win` source frames to `frames` with linear interpolation
                win = min(t, max(self.frames // 2, int(round(self.frames * random.uniform(*self.speed)))))
                start = random.uniform(0, t - win)
                pos = np.linspace(start, start + win - 1, self.frames)
                lo = np.floor(pos).astype(int)
                hi = np.minimum(lo + 1, t - 1)
                w = (pos - lo).astype(np.float32)[:, None, None]
                clip = ((1 - w) * clip[lo] + w * clip[hi]).round().astype(np.uint8)  # no repeated/skipped frames
            elif t < self.frames:  # rare short clip: loop it up to the window length
                clip = clip[np.arange(self.frames) % t]
            else:
                start = random.randint(0, t - self.frames)
                clip = clip[start:start + self.frames]
        return {
            "clip": torch.from_numpy(np.ascontiguousarray(clip)),
            "driver": it["driver_idx"], "source": it["source_idx"],
            "mode": it.get("mode", "speaking"), "set": it.get("set", "gallery"),
            "relpath": it["relpath"],
        }


class DriverSourceSampler(Sampler[list[int]]):
    """Each batch: p drivers x ps shared sources x k clips (same-source/different-driver hard negatives)."""

    def __init__(self, ds: CropDataset, p: int, ps: int, k: int, iterations: int, seed: int, mode_balance: bool = False) -> None:
        self.p, self.ps, self.k, self.iterations = p, ps, k, iterations
        self.mode_balance = mode_balance
        self.by_pair_mode: dict[tuple[int, int, str], list[int]] = defaultdict(list)
        self.rng = np.random.default_rng(seed)
        self.by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.sources_of: dict[int, set[int]] = defaultdict(set)
        for i, it in enumerate(ds.items):
            self.by_pair[it["driver_idx"], it["source_idx"]].append(i)
            self.by_pair_mode[it["driver_idx"], it["source_idx"], it["mode"]].append(i)
            self.sources_of[it["driver_idx"]].add(it["source_idx"])
        self.drivers = sorted(self.sources_of)
        if len(self.drivers) < 2:
            raise RuntimeError("Need at least 2 drivers in the training cache.")

    def _pick(self, values: list[int], n: int) -> list[int]:
        return self.rng.choice(values, size=n, replace=len(values) < n).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.iterations):
            drivers = self.rng.choice(self.drivers, size=min(self.p, len(self.drivers)), replace=False).tolist()
            common = sorted(set.intersection(*(self.sources_of[d] for d in drivers)))
            if len(common) >= self.ps:
                sources = self.rng.choice(common, size=self.ps, replace=False).tolist()
                pairs = [(d, s) for d in drivers for s in sources]
            else:  # partial cache: each driver draws its own sources
                pairs = [(d, s) for d in drivers
                         for s in self._pick(sorted(self.sources_of[d]), self.ps)]
            if self.mode_balance:                      # alternate speaking / listening within each pair
                yield [self._pick(self.by_pair_mode[d, s, m] or self.by_pair[d, s], 1)[0]
                       for d, s in pairs for m in (('speaking', 'listening') * self.k)[:self.k]]
            else:
                yield [i for pair in pairs for i in self._pick(self.by_pair[pair], self.k)]

    def __len__(self) -> int:
        return self.iterations


# ---------------------------------------------------------------------------
# Augmentation (GPU). Every random parameter is drawn per clip and shared by all frames of that clip,
# so the motion inside a clip is never distorted, only how it is rendered.
# ---------------------------------------------------------------------------

def _gauss_blur(x: Tensor, sigma: float) -> Tensor:
    """Separable depthwise Gaussian blur of [B, T, H, W] (T acts as the channel axis)."""
    radius = max(1, int(round(3 * sigma)))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    t = x.shape[1]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(t, 1, 1, -1), groups=t)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1).expand(t, 1, -1, 1), groups=t)


def augment_batch(clips: Tensor, cfg: Settings) -> Tensor:
    """clips [B, T, S, S] uint8 (GPU) -> float [B, T, in, in] in 0..255 with renderer-style perturbations.
    cfg.aug_parts selects the active ones (amp, photo, blur, spatial, noise; "speed" acts in the dataset) and
    cfg.aug_strength scales every perturbation towards the identity (1.0 = full range)."""
    b, t, s, _ = clips.shape
    dev = clips.device
    parts = set(cfg.aug_parts.split(","))
    st = cfg.aug_strength

    def u(lo: float, hi: float, ident: float = 1.0) -> Tensor:
        return torch.empty(b, 1, 1, 1, device=dev).uniform_(ident + st * (lo - ident), ident + st * (hi - ident))

    x = clips.float() / 255.0
    if "amp" in parts:                                   # exaggerate / damp movement around the clip's temporal mean
        mean = x.mean(1, keepdim=True)
        x = mean + u(0.8, 1.2) * (x - mean)
    if "photo" in parts:                                 # gamma, contrast, brightness
        x = x.clamp(0, 1) ** u(0.75, 1.35)
        x = (x - 0.5) * u(0.75, 1.3) + 0.5 + u(-0.08, 0.08, 0.0)
    if "blur" in parts:                                  # 0-1: sharp, 2: mild, 3: strong (chance scaled by strength)
        choice = torch.randint(0, 4, (b, 1, 1, 1), device=dev)
        choice = torch.where(torch.rand(b, 1, 1, 1, device=dev) < st, choice, torch.zeros_like(choice))
        x = torch.where(choice == 2, _gauss_blur(x, 0.6), x)
        x = torch.where(choice == 3, _gauss_blur(x, 1.0), x)
    out = cfg.input_size or s
    if "spatial" in parts:                               # zoom in up to 15 %, rotate +-6 deg, shift, resample
        scale = u(0.85, 1.0).view(b)
        ang = torch.empty(b, device=dev).uniform_(-6, 6) * st * math.pi / 180
        tx = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale + 0.03 * st)
        ty = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale + 0.03 * st)
        theta = torch.stack([torch.stack([scale * torch.cos(ang), -scale * torch.sin(ang), tx], 1),
                             torch.stack([scale * torch.sin(ang), scale * torch.cos(ang), ty], 1)], 1)
        grid = F.affine_grid(theta, (b, t, out, out), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    elif out != s:
        x = F.interpolate(x, size=(out, out), mode="area")
    if "noise" in parts:   # fixed pattern identical in every frame (cancels in differences) + tiny per-frame part
        x = x + torch.randn(b, 1, out, out, device=dev) * u(0.0, 0.03, 0.0)
        x = x + torch.randn_like(x) * u(0.0, 0.004, 0.0)
    return x.clamp(0, 1) * 255.0


def deflicker(x: Tensor) -> Tensor:
    """[B, T, S, S] float -> each frame rescaled to the clip's own mean/std (scale clamped to [0.5, 2]).
    Removes global per-frame brightness/contrast flicker, which differencing would otherwise read as motion."""
    fm = x.mean(dim=(-2, -1), keepdim=True)
    fs = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    cm = x.mean(dim=(1, 2, 3), keepdim=True)
    cs = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x - fm) * (cs / fs).clamp(0.5, 2.0) + cm


def mixstyle(f: Tensor, b: int, t: int, alpha: float = 0.1) -> Tensor:
    """MixStyle with per-clip statistics: mixes channel mean/std (over time and space) between random clips of the
    batch with one Beta-sampled weight per clip, so every frame of a clip changes consistently (no flicker)."""
    _, c, h, w = f.shape
    x = f.float().view(b, t, c, h, w)
    mu = x.mean(dim=(1, 3, 4), keepdim=True)
    sig = (x.var(dim=(1, 3, 4), keepdim=True, unbiased=False) + 1e-5).sqrt()
    perm = torch.randperm(b, device=f.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((b, 1, 1, 1, 1)).to(f.device)
    x = (x - mu) / sig * (lam * sig + (1 - lam) * sig[perm]) + (lam * mu + (1 - lam) * mu[perm])
    return x.view(b * t, c, h, w).to(f.dtype)


def uniformity_loss(z: Tensor, labels: Tensor, t: float = 2.0) -> Tensor:
    """Spread embeddings of *different* drivers over the sphere (they currently crowd into a narrow cone)."""
    z = F.normalize(z.float(), dim=-1)
    d2 = torch.cdist(z, z).pow(2)
    different = labels[:, None] != labels[None, :]
    return torch.log(torch.exp(-t * d2[different]).mean())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GlobalRowCol(nn.Module):
    """Depthwise circular conv spanning a whole row / column (global receptive field)."""

    def __init__(self, dim: int, size: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.BatchNorm2d(dim)
        self.wh = nn.Parameter(torch.randn(dim, 1, size, 1) * 0.02)
        self.ww = nn.Parameter(torch.randn(dim, 1, 1, size) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = F.conv2d(torch.cat([y, y[:, :, :-1, :]], 2), self.wh.to(y.dtype), groups=self.dim)
        y = F.conv2d(torch.cat([y, y[:, :, :, :-1]], 3), self.ww.to(y.dtype), groups=self.dim)
        return x + y


class ChannelGraph(nn.Module):
    """kNN over spatial positions (cosine similarity in channel space), max-aggregated differences."""

    def __init__(self, k: int = 4) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        pos = h * w
        feat = x.reshape(n, c, pos)
        normed = F.normalize(feat, dim=1)
        idx = torch.bmm(normed.transpose(1, 2), normed).topk(min(self.k, pos), dim=-1).indices  # (N, pos, k)
        tokens = feat.transpose(1, 2)                                                          # (N, pos, C)
        base = torch.arange(n, device=x.device).view(-1, 1, 1) * pos
        neighbours = tokens.reshape(n * pos, c)[(idx + base).reshape(-1)].view(n, pos, -1, c)
        edge = (neighbours - tokens.unsqueeze(2)).max(dim=2).values
        return F.relu(tokens + edge).transpose(1, 2).reshape(n, c, h, w)


class MotionEncoder(nn.Module):
    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self.variant = cfg.variant
        if self.variant not in ("pixdiff", "featdiff"):
            raise ValueError(f"unknown variant {self.variant!r}")
        self.lags = [int(v) for v in cfg.diff_lags.split(",")]
        if self.variant == "featdiff" and self.lags != [1]:
            raise ValueError("diff_lags other than '1' are only supported for the pixdiff variant")

        def c(n: int) -> int:                                    # channel count scaled by the width multiplier
            return max(4, int(round(n * cfg.width)))

        self.frame_enc = nn.Sequential(
            nn.Conv2d(len(self.lags), c(8), 4, 2), nn.BatchNorm2d(c(8)), nn.ReLU(True),
            nn.Conv2d(c(8), c(32), 3, 2), nn.BatchNorm2d(c(32)), nn.ReLU(True),
            nn.Conv2d(c(32), c(64), 2, 2, 1), nn.BatchNorm2d(c(64)), nn.ReLU(True),
            nn.Conv2d(c(64), c(128), 1), nn.BatchNorm2d(c(128)), nn.ReLU(True),
        )
        self.input_size = cfg.input_size or cfg.image_size
        self.mixstyle_p = cfg.mixstyle
        self.deflicker = cfg.deflicker
        self.grad_ckpt = cfg.grad_checkpoint
        size = (self.input_size - 4) // 2 + 1
        size = (size - 3) // 2 + 1
        size = size // 2 + 1
        self.mix = GlobalRowCol(c(128), size) if cfg.use_mix else nn.Identity()
        self.graph = ChannelGraph(4) if cfg.use_graph else nn.Identity()
        self.head = nn.Sequential(
            nn.Conv3d(c(128), c(64), 3, (1, 1, 1), 1), nn.BatchNorm3d(c(64)), nn.ReLU(True),
            nn.Conv3d(c(64), c(32), 3, (1, 2, 1), 1), nn.BatchNorm3d(c(32)), nn.ReLU(True),
            nn.AdaptiveAvgPool3d((4, 4, 1)),
        )
        self.attn = nn.Conv3d(c(32), 1, 1) if cfg.temporal_pool == "attn" else None  # learned weights over time
        self.fc = nn.Sequential(
            nn.Linear(c(32) * 16, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, cfg.embedding_dim)
        )
        # centres/scales the embedding with statistics learned on TRAINING data only (frozen at test time)
        self.embed_bn = nn.BatchNorm1d(cfg.embedding_dim, affine=False) if cfg.embed_bn else nn.Identity()
        # Preserve both original parameter initialization and the control's RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.temporal_branch = RegionalTemporalBranch(c(32), c(32) * 16) if cfg.temporal_branch else None

    def forward(self, clips: Tensor) -> Tensor:
        """clips: [B, T, S, S] uint8 (or float 0-255) -> L2-normalised embeddings [B, D]."""
        x = clips.float()
        if x.shape[-1] != self.input_size:                       # cached crops larger than the network input
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="area")
        if self.deflicker:
            x = deflicker(x)
        x = x.div(255.0).unsqueeze(2)                            # [B, T, 1, S, S]
        if self.variant == "pixdiff":                            # one channel per difference lag: x[t+k] - x[t]
            top, n = max(self.lags), x.shape[1]
            x = torch.cat([x[:, k:n - top + k] - x[:, :n - top] for k in self.lags], dim=2)
        b, t = x.shape[:2]
        ck = self.grad_ckpt and self.training

        def seg(fn, inp):                                        # optionally recompute this stage in the backward pass
            return torch_ckpt.checkpoint(fn, inp, use_reentrant=False) if ck else fn(inp)

        f = seg(self.frame_enc[:6], x.flatten(0, 1))
        if self.training and self.mixstyle_p > 0 and random.random() < self.mixstyle_p:
            f = mixstyle(f, b, t)
        f = seg(lambda z: self.graph(self.mix(self.frame_enc[6:](z))), f)
        f = f.view(b, t, *f.shape[1:])                           # [B, T', C, H, W]
        if self.variant == "featdiff":
            f = f[:, 1:] - f[:, :-1]
        vol = f.permute(0, 2, 3, 4, 1)                           # [B, C, H, W, T'']
        h = seg(self.head[:6], vol)
        temporal = self.temporal_branch(h) if self.temporal_branch is not None else None
        if self.attn is not None:                                # attention over time instead of a plain average
            w = torch.softmax(self.attn(h.mean(dim=(2, 3), keepdim=True)), dim=-1)
            h = (h * w).sum(dim=-1, keepdim=True)
        pooled = self.head[6](h).flatten(1)
        if temporal is not None:
            pooled = pooled + temporal
        return F.normalize(self.embed_bn(self.fc(pooled)), dim=-1)


class MotionEncoder3D(nn.Module):
    """Structurally different from MotionEncoder: 3D convolutions mix time and space from the FIRST layer and keep
    downsampling time throughout (instead of one late, narrow-kernel 3D head), for a different inductive bias and
    genuine ensemble diversity. Same input/output contract, embed_bn, dropout, width scaling as MotionEncoder."""

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self.variant = cfg.variant
        if self.variant not in ("pixdiff", "featdiff"):
            raise ValueError(f"unknown variant {self.variant!r}")
        self.lags = [int(v) for v in cfg.diff_lags.split(",")]
        if self.variant == "featdiff":
            raise ValueError("MotionEncoder3D only supports the pixdiff variant (differencing must happen before the 3D stem)")

        def c(n: int) -> int:
            return max(4, int(round(n * cfg.width)))

        self.input_size = cfg.input_size or cfg.image_size
        self.mixstyle_p = cfg.mixstyle
        self.deflicker = cfg.deflicker
        self.grad_ckpt = cfg.grad_checkpoint
        self.stem = nn.Sequential(                                       # early fusion: time and space mixed together
            nn.Conv3d(len(self.lags), c(16), (3, 4, 4), (1, 2, 2), (1, 0, 0)), nn.BatchNorm3d(c(16)), nn.ReLU(True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(c(16), c(32), 3, (2, 2, 2), 1), nn.BatchNorm3d(c(32)), nn.ReLU(True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(c(32), c(64), 3, (2, 2, 2), 1), nn.BatchNorm3d(c(64)), nn.ReLU(True),
            nn.Conv3d(c(64), c(128), 1), nn.BatchNorm3d(c(128)), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(c(128), 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, cfg.embedding_dim)
        )
        self.embed_bn = nn.BatchNorm1d(cfg.embedding_dim, affine=False) if cfg.embed_bn else nn.Identity()

    def forward(self, clips: Tensor) -> Tensor:
        """clips: [B, T, S, S] uint8 (or float 0-255) -> L2-normalised embeddings [B, D]."""
        x = clips.float()
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="area")
        if self.deflicker:
            x = deflicker(x)
        x = x.div(255.0).unsqueeze(2)                                    # [B, T, 1, S, S]
        top, n = max(self.lags), x.shape[1]
        x = torch.cat([x[:, k:n - top + k] - x[:, :n - top] for k in self.lags], dim=2)   # [B, T', L, S, S]
        b, t = x.shape[:2]
        vol = x.permute(0, 2, 1, 3, 4)                                   # [B, L, T', S, S] (channels first for Conv3d)
        ck = self.grad_ckpt and self.training

        def seg(fn, inp):
            return torch_ckpt.checkpoint(fn, inp, use_reentrant=False) if ck else fn(inp)

        h = seg(self.stem, vol)
        if self.training and self.mixstyle_p > 0 and random.random() < self.mixstyle_p:
            h = mixstyle3d(h)
        h = seg(self.stage2, h)
        h = seg(self.stage3, h)
        return F.normalize(self.embed_bn(self.fc(self.pool(h).flatten(1))), dim=-1)


def mixstyle3d(f: Tensor, p: float = 0.5, eps: float = 1e-6) -> Tensor:
    """MixStyle for a 5D [N, C, T, H, W] feature map: mixes per-clip channel statistics with another random clip."""
    n = f.shape[0]
    if n < 2:
        return f
    mu = f.mean(dim=(2, 3, 4), keepdim=True)
    sig = f.std(dim=(2, 3, 4), keepdim=True) + eps
    perm = torch.randperm(n, device=f.device)
    lam = torch.rand(n, 1, 1, 1, 1, device=f.device) * 0.5
    mix_mu = lam * mu + (1 - lam) * mu[perm]
    mix_sig = lam * sig + (1 - lam) * sig[perm]
    return (f - mu) / sig * mix_sig + mix_mu


def build_model(cfg: Settings) -> nn.Module:
    if cfg.arch == "3d" and cfg.temporal_branch:
        raise ValueError("temporal_branch is only implemented for the 2D encoder")
    return MotionEncoder3D(cfg) if cfg.arch == "3d" else MotionEncoder(cfg)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def cosine_with_warmup(opt: torch.optim.Optimizer, warmup: int, total: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def supcon_loss(z: Tensor, labels: Tensor, temperature: float) -> Tensor:
    """Supervised contrastive loss; positives = clips of the same driver (any avatar)."""
    z = F.normalize(z.float(), dim=-1)
    logits = (z @ z.T) / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    count = pos.sum(1)
    valid = count > 0
    per_anchor = log_prob.masked_fill(~pos, 0).sum(1)[valid] / count[valid]
    return -per_anchor.mean()


def identity_loss(z: Tensor, labels: Tensor, modes: Tensor | None, cfg: Settings) -> Tensor:
    base = supcon_loss(z, labels, cfg.temperature)
    if cfg.cross_mode_weight < 0:
        raise ValueError("cross_mode_weight must be nonnegative")
    if cfg.cross_mode_weight:
        if modes is None:
            raise ValueError("cross-mode loss requires training mode labels")
        cross = cross_mode_positive_loss(z, labels, modes, cfg.temperature)
        if cross is not None:
            base = (base + cfg.cross_mode_weight * cross) / (1 + cfg.cross_mode_weight)
    return base


def compute_auc(scores: Tensor, targets: Tensor) -> float:
    pos, neg = int(targets.sum()), int((1 - targets).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    t = targets[scores.argsort(descending=True)]
    tpr, fpr = t.cumsum(0).float() / pos, (1 - t).cumsum(0).float() / neg
    zero = torch.zeros(1, device=tpr.device)
    return torch.trapz(torch.cat((zero, tpr)), torch.cat((zero, fpr))).item()


def compute_eer(scores: Tensor, targets: Tensor) -> float:
    pos, neg = int(targets.sum()), int((1 - targets).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    t = targets[scores.argsort(descending=True)]
    tpr, fpr = t.cumsum(0).float() / pos, (1 - t).cumsum(0).float() / neg
    diff = fpr - (1 - tpr)
    idx = (diff[:-1] * diff[1:] <= 0).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return (fpr[-1].item() + (1 - tpr[-1]).item()) / 2
    i = idx[0].item()
    d0, d1 = diff[i].item(), diff[i + 1].item()
    a = d0 / (d0 - d1) if d0 != d1 else 0.5
    return fpr[i].item() * (1 - a) + fpr[i + 1].item() * a


def pair_metrics(pe: Tensor, pid: Tensor, ge: Tensor, gid: Tensor, metric: str) -> tuple[float, float]:
    if metric == "euclidean":
        sim = -torch.cdist(pe, ge)
    else:
        sim = F.normalize(pe, dim=-1) @ F.normalize(ge, dim=-1).T
    targets = (pid[:, None] == gid[None, :]).flatten().float()
    return compute_auc(sim.flatten(), targets), compute_eer(sim.flatten(), targets)


def official_score(m: dict[str, float]) -> float:
    """Repo convention: mean of speaking, listening and cross (mean of the two cross pairs)."""
    g = lambda k: m.get(f"emb_{k}_auc", float("nan"))  # noqa: E731
    cross = (g("speaking_vs_listening") + g("listening_vs_speaking")) / 2
    vals = [v for v in (g("speaking_vs_speaking"), g("listening_vs_listening"), cross) if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else 0.0


def mean4(m: dict[str, float]) -> float:
    vals = [m.get(f"emb_{a}_vs_{b}_auc", float("nan")) for a in ("speaking", "listening") for b in ("speaking", "listening")]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else 0.0


EVAL_TTA = 1   # number of test-time views per window (1 = off); set with --tta
_VIEWS = [(1.0, 0.0, 0.0), (0.93, 0.0, 0.0), (0.97, 0.03, 0.0), (0.97, -0.03, 0.0), (0.97, 0.0, 0.03), (0.97, 0.0, -0.03)]


EVAL_TTA_MODE = "geom"   # "geom" (zoom/shift), "render" (renderer-like), or "both"; set with --tta-mode


def _render_views(windows: Tensor) -> list[Tensor]:
    """Renderer-style test-time views: soften / sharpen (opposite ends of the sharpness axis, so
    averaging over them cancels the model's sensitivity to it) plus a fixed-pattern noise pair.
    Deterministic, per-clip, no test-set statistics involved.

    Deliberately disjoint from the robustness-proxy profiles (JPEG, temporal smoothing, 2x
    down-up, tone curve) so that scoring this on the proxy stays an honest held-out test.
    """
    x = windows.float()
    b, t, h, w = x.shape
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device)
    k = (k / k.sum()).view(1, 1, -1, 1)
    flat = x.reshape(b * t, 1, h, w)
    blur = F.conv2d(F.pad(flat, (0, 0, 2, 2), mode="replicate"), k)
    blur = F.conv2d(F.pad(blur, (2, 2, 0, 0), mode="replicate"), k.transpose(-1, -2)).reshape(b, t, h, w)
    sharp = (x + 0.8 * (x - blur)).clamp(0, 255)
    grain = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, 1, h, 1) *             torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, 1, w) * 3.0
    return [blur, sharp, (x + grain).clamp(0, 255), (x - grain).clamp(0, 255)]


def _views(windows: Tensor, n: int) -> list[Tensor]:
    """n deterministic views of [W, T, S, S]: the windows themselves plus slight zoom/shift variants."""
    out = []
    if EVAL_TTA_MODE in ("render", "both"):
        out.append(windows)
        out.extend(_render_views(windows)[:max(0, n - 1)])
        if EVAL_TTA_MODE == "render":
            return out
        out = out[:max(1, n)]
    for s, dx, dy in _VIEWS[:max(1, n)]:
        if (s, dx, dy) == (1.0, 0.0, 0.0):
            out.append(windows)
            continue
        theta = torch.tensor([[s, 0.0, dx], [0.0, s, dy]], device=windows.device).expand(windows.shape[0], 2, 3)
        grid = F.affine_grid(theta, tuple(windows.shape), align_corners=False)
        out.append(F.grid_sample(windows.float(), grid, mode="bilinear", padding_mode="border", align_corners=False))
    return out


@torch.no_grad()
def embed_clip(model: nn.Module, clip: Tensor, device: torch.device, frames: int, stride: int) -> Tensor:
    """Sliding-window embedding of one variable-length clip [T, S, S]; windows (and test-time views) are averaged."""
    t = clip.shape[0]
    if t <= frames:
        windows = clip[None]
    else:
        starts = list(range(0, t - frames + 1, stride))
        if starts[-1] != t - frames:
            starts.append(t - frames)
        windows = torch.stack([clip[s:s + frames] for s in starts])
    windows = windows.to(device, non_blocking=True)
    emb = torch.cat([model(v) for v in _views(windows, EVAL_TTA)])
    return F.normalize(emb.float().mean(0, keepdim=True), dim=-1)[0]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: Settings) -> dict[str, float]:
    model.eval()
    buckets: dict[tuple[str, str], tuple[list[Tensor], list[int]]] = defaultdict(lambda: ([], []))
    for batch in loader:  # batch_size = 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            emb = embed_clip(model, batch["clip"][0], device, cfg.frames, cfg.window_stride)
        key = (batch["set"][0], batch["mode"][0])
        buckets[key][0].append(emb.cpu())
        buckets[key][1].append(int(batch["driver"][0]))
    out: dict[str, float] = {}
    for pm in ("speaking", "listening"):
        for gm in ("speaking", "listening"):
            p, g = buckets.get(("probe", pm)), buckets.get(("gallery", gm))
            if not p or not g or not p[0] or not g[0]:
                continue
            auc, eer = pair_metrics(torch.stack(p[0]), torch.tensor(p[1]), torch.stack(g[0]), torch.tensor(g[1]),
                                    cfg.eval_distance)
            out[f"emb_{pm}_vs_{gm}_auc"], out[f"emb_{pm}_vs_{gm}_eer"] = auc, eer
    out["score"], out["score4"] = official_score(out), mean4(out)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

SHIFT_PROFILES = ("jpeg", "smooth", "downup", "tone", "combo")


def _shift(clip: Tensor, profile: str) -> Tensor:
    """Renderer-like degradations of a [T, S, S] uint8 clip. None of them is used in training augmentation."""
    def smooth(x: Tensor) -> Tensor:            # temporal smoothing: mean of 3 neighbouring frames
        p = torch.cat([x[:1], x, x[-1:]])
        return (p[:-2] + p[1:-1] + p[2:]) / 3

    def downup(x: Tensor) -> Tensor:            # lose fine detail: 2x area downsample, bilinear upsample
        return F.interpolate(F.interpolate(x[None], scale_factor=0.5, mode="area"), size=x.shape[-2:],
                             mode="bilinear", align_corners=False)[0]

    def tone(x: Tensor) -> Tensor:              # different tone curve: gamma 1.5, lower contrast, brighter
        return (((x / 255.0).clamp(0, 1) ** 1.5 - 0.5) * 0.7 + 0.55).clamp(0, 1) * 255.0

    if profile == "jpeg":
        import cv2
        frames = [cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_GRAYSCALE)
                  for f in clip.numpy()]
        return torch.from_numpy(np.stack(frames))
    x = clip.float()
    if profile == "smooth":
        x = smooth(x)
    elif profile == "downup":
        x = downup(x)
    elif profile == "tone":
        x = tone(x)
    elif profile == "combo":
        x = tone(downup(smooth(x)))
    else:
        raise ValueError(profile)
    return x.round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def evaluate_shift(model: nn.Module, ds: CropDataset, device: torch.device, cfg: Settings) -> dict[str, float]:
    """Stand-in for the (unmeasurable) different-generator score: probe = clean clip, gallery = the same kind of clip
    after a renderer-like degradation. Returns the four-pair mean AUC per profile and their mean."""
    model.eval()
    modes = ("speaking", "listening")
    bucket: dict[tuple[str, str], list] = defaultdict(lambda: [[], [], {p: [] for p in SHIFT_PROFILES}])
    for i in range(len(ds)):
        item = ds[i]
        clip, key = item["clip"], (item["set"], item["mode"])
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            emb = embed_clip(model, clip, device, cfg.frames, cfg.window_stride)
            b = bucket[key]
            b[0].append(emb.cpu())
            b[1].append(item["driver"])
            if item["set"] == "gallery":
                for p in SHIFT_PROFILES:
                    b[2][p].append(embed_clip(model, _shift(clip, p), device, cfg.frames, cfg.window_stride).cpu())
    out: dict[str, float] = {}
    for p in SHIFT_PROFILES:
        aucs = []
        for pm in modes:
            for gm in modes:
                probe, gallery = bucket.get(("probe", pm)), bucket.get(("gallery", gm))
                if not probe or not gallery or not probe[0] or not gallery[0]:
                    continue
                auc, _ = pair_metrics(torch.stack(probe[0]), torch.tensor(probe[1]), torch.stack(gallery[2][p]),
                                      torch.tensor(gallery[1]), cfg.eval_distance)
                aucs.append(auc)
        out[f"shift_{p}_score4"] = float(np.mean(aucs)) if aucs else float("nan")
    out["shift_mean_score4"] = float(np.nanmean([out[f"shift_{p}_score4"] for p in SHIFT_PROFILES]))
    return out


EVAL_FRAMES = 0   # >0: window length used at evaluation instead of the training value (set with --eval-frames)


def settings_from_checkpoint(ckpt: dict) -> Settings:
    """Rebuild the model/evaluation settings a checkpoint was trained with (older checkpoints lack newer keys)."""
    c = ckpt.get("config", {})
    return replace(
        Settings(), variant=c.get("variant", "featdiff"), use_mix=c.get("use_mix", "True") == "True",
        use_graph=c.get("use_graph", "True") == "True", image_size=int(c.get("image_size", 64)),
        input_size=int(c.get("input_size", 0)), embed_bn=c.get("embed_bn", "False") == "True",
        frames=EVAL_FRAMES or int(c.get("frames", 64)), embedding_dim=int(c.get("embedding_dim", 256)),
        window_stride=int(c.get("window_stride", 36)), diff_lags=c.get("diff_lags", "1"),
        deflicker=c.get("deflicker", "False") == "True", temporal_pool=c.get("temporal_pool", "avg"), width=float(c.get("width", 1.0)), arch=c.get("arch", "2d"), cache_root=Path(c["cache_root"]) if "cache_root" in c else Settings().cache_root,
        temporal_branch=str(c.get("temporal_branch", "False")) == "True",
    )


def log_study_record(cfg: Settings, run_name: str, last: dict[str, float], shift: dict[str, float],
                     checkpoint: Path | None = None) -> None:
    """Append one line per finished run to output/video/study_log.jsonl (source of the ablation tables)."""
    import datetime
    try:
        ck = torch.load(checkpoint or cfg.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    except Exception:
        return
    m = ck.get("metrics", {})
    keys = ("variant", "image_size", "input_size", "frames", "diff_lags", "temporal_pool", "augment", "aug_parts",
            "aug_strength", "mixstyle", "embed_bn", "uniformity", "temperature", "learning_rate", "iterations",
            "seed", "batch_drivers", "batch_sources", "cross_samples", "use_mix", "use_graph", "ema_decay", "width", "deflicker",
            "arch", "grad_cache", "mode_balance", "cross_mode_weight", "temporal_branch", "schedule_iterations", "io_threads")
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "run": run_name, "group": cfg.group,
           "tag": cfg.run_tag, "best_iteration": ck.get("iteration"), "best_score3": m.get("score"),
           "best_score4": m.get("score4"),
           "best_pairs": {k: v for k, v in m.items() if k.startswith("emb_") and k.endswith("_auc")},
           "final_score3": last.get("score"), "final_score4": last.get("score4"), "shift": shift,
           "config": {k: getattr(cfg, k) for k in keys}}
    path = PROJECT_ROOT / "output/video/study_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _mlflow(fn, *args, **kwargs) -> None:
    """Live MLflow logging must never crash a training run (e.g. a DagsHub network hiccup)."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        LOGGER.warning("MLflow call failed (%s) - training continues", exc)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")], force=True)


def make_val_loader(cfg: Settings) -> DataLoader | None:
    ds = CropDataset(cfg, cfg.val_driver_split, cfg.val_source_split, train=False)
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=min(cfg.workers, 4),
                      persistent_workers=cfg.workers > 0)


def save_checkpoint(path: Path, cfg: Settings, it: int, model: nn.Module, metrics: dict, tag: str) -> None:
    torch.save({"iteration": it, "model": model.state_dict(), "metrics": metrics, "weights": tag,
                "config": {k: str(v) for k, v in cfg.__dict__.items()}}, path)


def cached_step(model: nn.Module, clips: Tensor, driver: Tensor, cfg: Settings, scaler: Any, device: torch.device,
                modes: Tensor | None = None, extra_loss: Any = None) -> Tensor:
    """One GradCache step: loss on the full batch, gradients accumulated chunk by chunk. Returns the (unscaled) loss."""
    chunks = list(clips.split(cfg.grad_cache))
    seeds = [random.getrandbits(31) for _ in chunks]
    cuda = [device] if device.type == "cuda" else []

    def embed(chunk: Tensor, seed: int) -> Tensor:
        state = random.getstate()
        try:
            with torch.random.fork_rng(devices=cuda):          # same dropout / MixStyle draws in both passes
                torch.manual_seed(seed)
                random.seed(seed)
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    return model(chunk).float()
        finally:
            random.setstate(state)

    with torch.no_grad():
        emb = torch.cat([embed(c, s) for c, s in zip(chunks, seeds)]).detach().requires_grad_(True)
    loss = identity_loss(emb, driver, modes, cfg)
    if cfg.uniformity > 0:
        loss = loss + cfg.uniformity * uniformity_loss(emb, driver)
    if extra_loss is not None:                                 # e.g. the auxiliary landmark term
        loss = loss + extra_loss(emb)
    scaler.scale(loss).backward()                              # gradient of the loss w.r.t. every embedding
    grads = emb.grad.split(cfg.grad_cache)
    for c, s, g in zip(chunks, seeds, grads):
        embed(c, s).backward(g)
    return loss.detach()


def audit_training_data(cfg: Settings, train_ds: CropDataset, val_ds: CropDataset | None) -> dict:
    """Fail closed on the provided metadata/splits; record inputs without secrets."""
    root = PROJECT_ROOT / "data"
    required = {
        "meta_path": root / "meta/train-mp4-meta.json",
        "train_driver_split": SPLIT_DIR / "driver_train_splits.json",
        "train_source_split": SPLIT_DIR / "source_train_splits.json",
        "val_driver_split": SPLIT_DIR / "driver_val_splits.json",
        "val_source_split": SPLIT_DIR / "source_val_splits.json",
    }
    for field, path in required.items():
        if getattr(cfg, field).resolve() != path.resolve():
            raise ValueError(f"Training requires the provided {field}")
    if cfg.cache_root.resolve().parent != (root / "video_crops").resolve() or not cfg.cache_root.name.startswith("train_"):
        raise ValueError("Training cache must be data/video_crops/train_*")
    splits = {k: set(json.loads(p.read_text())) for k, p in required.items() if k != "meta_path"}
    for axis in ("driver", "source"):
        if splits[f"train_{axis}_split"] & splits[f"val_{axis}_split"]:
            raise ValueError(f"Train/validation {axis} overlap")
    metadata = json.loads(cfg.meta_path.read_text())
    for name, ds in (("train", train_ds), ("val", val_ds)):
        expected = {m["relpath"] for m in metadata
                    if m["driver_id"] in splits[f"{name}_driver_split"]
                    and m["source_id"] in splits[f"{name}_source_split"]}
        if ds is None or len(ds) != len(expected) or {m["relpath"] for m in ds.items} != expected:
            raise ValueError(f"Incomplete {name} cache; expected {len(expected)} provided clips")
    return {"initialization": "random", "external_data_or_weights": False, "test_data_used": False,
            "training_clips": len(train_ds), "validation_clips": len(val_ds),
            "cache": str(cfg.cache_root.relative_to(PROJECT_ROOT)),
            "sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in required.values()}}


def train(cfg: Settings) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    if (cfg.output_dir / "best_model.pt").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {cfg.output_dir}")
    setup_logging(cfg.output_dir)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CropDataset(cfg, cfg.train_driver_split, cfg.train_source_split, train=True)
    sampler = DriverSourceSampler(train_ds, cfg.batch_drivers, cfg.batch_sources, cfg.cross_samples,
                                  cfg.iterations, cfg.seed, cfg.mode_balance)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.workers, pin_memory=True,
                        persistent_workers=cfg.workers > 0, generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = make_val_loader(cfg)
    if val_loader is None:
        LOGGER.warning("validation set is empty in the current cache - validation will be skipped")

    provenance = audit_training_data(cfg, train_ds, val_loader.dataset if val_loader else None)
    (cfg.output_dir / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    model = build_model(cfg).to(device)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    LOGGER.info("variant=%s | params %.3fM | device %s", cfg.variant,
                sum(p.numel() for p in model.parameters()) / 1e6, device)
    aux_proj, aux_targets, aux_rows = None, None, None
    if cfg.aux_landmark > 0:
        blob = torch.load(cfg.aux_landmark_path, map_location="cpu", weights_only=False)
        aux_rows = {k: i for i, k in enumerate(blob["keys"])}
        aux_targets = F.normalize(blob["emb"].float(), dim=-1).to(device)
        missing = [it["relpath"][:-4] for it in train_ds.items if it["relpath"][:-4] not in aux_rows]
        if missing:
            raise SystemExit(f"{len(missing)} training clips have no landmark target (e.g. {missing[:2]}); "
                             "run scripts/precompute_landmark_targets.py first")
        aux_proj = nn.Linear(cfg.embedding_dim, aux_targets.shape[1]).to(device)
        LOGGER.info("auxiliary landmark loss: weight %.3f | %d targets of dim %d from %s",
                    cfg.aux_landmark, aux_targets.shape[0], aux_targets.shape[1], cfg.aux_landmark_path.name)
    params = list(model.parameters()) + (list(aux_proj.parameters()) if aux_proj is not None else [])
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    horizon = cfg.schedule_iterations or cfg.iterations
    if horizon < cfg.iterations:
        raise ValueError("schedule_iterations must cover all training iterations")
    sched = cosine_with_warmup(opt, cfg.warmup_steps, horizon, cfg.min_lr / cfg.learning_rate)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    batch_size = cfg.batch_drivers * cfg.batch_sources * cfg.cross_samples
    # naming convention: one experiment for the video branch, run name = variant + resolution + batch size
    run_name = (f"mdnet-{cfg.variant}-{cfg.input_size or cfg.image_size}px-b{batch_size}"
                f"{'-aug' if cfg.augment else ''}{'-' + cfg.run_tag if cfg.run_tag else ''}")
    setup_mlflow_run("video-mdnet", run_name, cfg)
    mlflow.set_tags({"initialization": "random", "training_data": "provided_training_split_only",
                     "external_models": "none", "test_adaptation": "none"})
    mlflow.log_artifact(str(cfg.output_dir / "data_provenance.json"))
    (cfg.output_dir / "mlflow_run_id.txt").write_text(mlflow.active_run().info.run_id, encoding="utf-8")
    if cfg.group:
        _mlflow(mlflow.set_tag, "group", cfg.group)
    csv_logger = CSVMetricsLogger(cfg.output_dir)
    best, bad, t0 = -1.0, 0, time.time()
    last_metrics: dict[str, float] = {}
    try:
        for it, batch in enumerate(loader, 1):
            model.train()
            clips = batch["clip"].to(device, non_blocking=True)
            driver = batch["driver"].to(device, non_blocking=True)
            modes = torch.tensor([{"speaking": 0, "listening": 1}[m] for m in batch["mode"]], device=device)
            if cfg.augment:
                clips = augment_batch(clips, cfg)
            aux_fn, aux_value = None, None
            if aux_proj is not None:
                target = aux_targets[torch.tensor([aux_rows[r[:-4]] for r in batch["relpath"]], device=device)]

                def aux_fn(e, target=target):                  # cosine distance to the landmark embedding
                    nonlocal aux_value
                    pred = F.normalize(aux_proj(e.float()), dim=-1)
                    term = cfg.aux_landmark * (1.0 - (pred * target).sum(-1)).mean()
                    aux_value = float(term.detach())
                    return term

            cons_value = None
            if cfg.consistency > 0:
                # second view: same motion, different rendering. The batch becomes [clean ; perturbed], so the
                # identity loss also sees the pair as a positive and the halves stay aligned through GradCache.
                n_clean = clips.shape[0]
                perturbed = augment_batch(clips, replace(cfg, augment=True, aug_parts=cfg.consistency_parts,
                                                         aug_strength=cfg.consistency_strength))
                # augment_batch already resizes to the network input; match the clean view so the halves concatenate
                # (area interpolation here is exactly what the model would do internally, so the clean view is unchanged)
                out = perturbed.shape[-1]
                clean = clips.float() if clips.shape[-1] == out else F.interpolate(clips.float(), size=(out, out), mode="area")
                clips = torch.cat([clean, perturbed])
                driver = torch.cat([driver, driver])
                modes = torch.cat([modes, modes])

                def cons_fn(e, n=n_clean):
                    nonlocal cons_value
                    a = F.normalize(e[:n].float(), dim=-1)
                    b = F.normalize(e[n:].float(), dim=-1)
                    term = cfg.consistency * (1.0 - (a * b).sum(-1)).mean()
                    cons_value = float(term.detach())
                    return term
            else:
                cons_fn = None

            terms = [f for f in (aux_fn, cons_fn) if f is not None]
            extra = (lambda e: sum(f(e) for f in terms)) if terms else None

            opt.zero_grad(set_to_none=True)
            if cfg.grad_cache > 0:
                loss = cached_step(model, clips, driver, cfg, scaler, device, modes, extra)
            else:
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    emb = model(clips)
                loss = identity_loss(emb, driver, modes, cfg)
                if cfg.uniformity > 0:
                    loss = loss + cfg.uniformity * uniformity_loss(emb, driver)
                if extra is not None:
                    loss = loss + extra(emb)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update(model)
            if it == 1 or it % 50 == 0:
                lr = sched.get_last_lr()[0]
                LOGGER.info("iter %d/%d | loss %.4f | lr %.2e | %.2f s/it", it, cfg.iterations, loss.item(), lr,
                            (time.time() - t0) / it)
                metrics = {"train/loss": loss.item(), "train/lr": lr}
                if aux_value is not None:
                    metrics["train/aux_landmark"] = aux_value
                if cons_value is not None:
                    metrics["train/consistency"] = cons_value
                _mlflow(mlflow.log_metrics, metrics, step=it)
                csv_logger.log("train", {k.split("/")[-1]: v for k, v in metrics.items()}, step=it)
            if val_loader is not None and (it % cfg.validation_interval == 0 or it == cfg.iterations):
                results = {"live": evaluate(model, val_loader, device, cfg)}
                if ema is not None:
                    results["ema"] = evaluate(ema.model, val_loader, device, cfg)
                tag, metrics = max(results.items(), key=lambda kv: kv[1]["score"])
                last_metrics = metrics
                for name, m in results.items():
                    _mlflow(mlflow.log_metrics, {f"val/{name}_{k}": v for k, v in m.items() if not math.isnan(v)}, step=it)
                csv_logger.log("val", {f"{n}_{k}": v for n, m in results.items() for k, v in m.items()}, step=it)
                LOGGER.info("validation @%d | %s | score %.4f (4-pair %.4f) | " + " ".join(
                    f"{k.replace('emb_', '').replace('_auc', '')}={v:.3f}" for k, v in metrics.items() if k.endswith("_auc")),
                    it, tag, metrics["score"], metrics["score4"])
                if metrics["score"] > best:
                    best, bad = metrics["score"], 0
                    save_checkpoint(cfg.output_dir / "best_model.pt", cfg, it,
                                    (ema.model if tag == "ema" else model), metrics, tag)
                    _mlflow(mlflow.log_metric, "val/best_score", best, step=it)
                    LOGGER.info("new best (%s) saved: %.4f", tag, best)
                else:
                    bad += 1
                    if cfg.patience and bad >= cfg.patience:
                        LOGGER.info("early stopping after %d validations without improvement", bad)
                        break
        save_checkpoint(cfg.output_dir / "last_model.pt", cfg, it, model, {}, "live")
        shift: dict[str, float] = {}
        if cfg.shift_eval and val_loader is not None and (cfg.output_dir / "best_model.pt").is_file():
            try:  # robustness proxy on the best checkpoint
                best_ck = torch.load(cfg.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
                probe_model = build_model(cfg).to(device)
                probe_model.load_state_dict(best_ck["model"])
                shift = evaluate_shift(probe_model, val_loader.dataset, device, cfg)
                _mlflow(mlflow.log_metrics, {f"shift/{k}": v for k, v in shift.items()})
                LOGGER.info("robustness proxy (best checkpoint): " + " | ".join(f"{k}={v:.4f}" for k, v in shift.items()))
                del probe_model
            except Exception as exc:
                LOGGER.warning("shift evaluation failed (%s)", exc)
        log_study_record(cfg, run_name, last_metrics, shift)
        if cfg.register and (cfg.output_dir / "best_model.pt").is_file():
            try:  # a finished training must never be lost over an upload problem
                from register import finalize_active_run
                LOGGER.info(finalize_active_run(cfg.output_dir))
            except Exception as exc:
                LOGGER.warning("MLflow model upload failed (%s). Run: python scripts/register_video.py %s",
                               exc, cfg.output_dir)
    finally:
        train_ds.close()
        csv_logger.close()
        mlflow.end_run()


def test(cfg: Settings, checkpoint: Path) -> None:
    setup_logging(cfg.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    val_loader = make_val_loader(cfg)
    if val_loader is None:
        raise RuntimeError("validation set is empty in the cache")
    m = evaluate(model, val_loader, device, cfg)
    LOGGER.info("score %.6f | 4-pair mean %.6f | %s", m["score"], m["score4"],
                {k: round(v, 4) for k, v in m.items() if k.endswith("_auc")})


def shift_only(checkpoint: Path, group: str = "shift-reference", tag: str | None = None) -> None:
    """Score an existing checkpoint on the clean validation set and on the synthetic-shift proxy."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tag = tag or checkpoint.parent.name
    cfg = replace(settings_from_checkpoint(ckpt), output_dir=PROJECT_ROOT / "output/video/shift_ref" / tag,
                  group=group, run_tag=tag)
    setup_logging(cfg.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    val_loader = make_val_loader(cfg)
    clean = evaluate(model, val_loader, device, cfg)
    shift = evaluate_shift(model, val_loader.dataset, device, cfg)
    LOGGER.info("%s | tta %d | clean 3-cond %.4f 4-pair %.4f | %s", tag, EVAL_TTA, clean["score"], clean["score4"],
                " | ".join(f"{k}={v:.4f}" for k, v in shift.items()))
    log_study_record(cfg, tag, clean, shift, checkpoint)


@torch.no_grad()
def ensemble_only(checkpoints: list[Path], group: str, tag: str) -> None:
    """Ensemble of several video checkpoints on the validation set: the similarity of two clips is the mean of the
    cosine similarities of the individual models (same modality, no test statistics). Also prints each member alone."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embs: list[dict[str, Tensor]] = []
    info: dict[str, tuple[str, str, str]] = {}
    names: list[str] = []
    for path in checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = settings_from_checkpoint(ck)
        model = build_model(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        ds = CropDataset(cfg, cfg.val_driver_split, cfg.val_source_split, train=False)
        e: dict[str, Tensor] = {}
        for i in range(len(ds)):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                v = embed_clip(model, ds[i]["clip"], device, cfg.frames, cfg.window_stride)
            key = ds.items[i]["relpath"]
            e[key] = F.normalize(v.float().cpu(), dim=-1)
            info.setdefault(key, (ds.items[i]["set"], ds.items[i]["mode"], ds.items[i]["driver_id"]))
        embs.append(e)
        names.append(path.parent.name)
        del model
        torch.cuda.empty_cache()
    keys = sorted(set.intersection(*[set(e) for e in embs]))
    mats = [torch.stack([e[k] for k in keys]) for e in embs]
    drivers = {d: i for i, d in enumerate(sorted({info[k][2] for k in keys}))}
    lab = torch.tensor([drivers[info[k][2]] for k in keys])
    modes = ("speaking", "listening")
    idx = {(s, m): torch.tensor([i for i, k in enumerate(keys) if info[k][0] == s and info[k][1] == m])
           for s in ("probe", "gallery") for m in modes}

    def scores(sim: Tensor) -> tuple[float, float, dict[str, float]]:
        pairs = {}
        for pm in modes:
            for gm in modes:
                p, g = idx[("probe", pm)], idx[("gallery", gm)]
                pairs[f"emb_{pm}_vs_{gm}_auc"] = compute_auc(
                    sim[p][:, g].flatten(), (lab[p][:, None] == lab[g][None, :]).flatten().float())
        return official_score(pairs), mean4(pairs), pairs

    for name, m in zip(names, mats):
        s3, s4, _ = scores(m @ m.T)
        LOGGER.info("member %-24s 3-cond %.4f 4-pair %.4f", name, s3, s4)
    s3, s4, pairs = scores(sum(m @ m.T for m in mats) / len(mats))
    LOGGER.info("ENSEMBLE of %d (tta %d): 3-cond %.4f | 4-pair %.4f | %s", len(mats), EVAL_TTA, s3, s4,
                {k.replace("emb_", "").replace("_auc", ""): round(v, 4) for k, v in pairs.items()})
    path = PROJECT_ROOT / "output/video/study_log.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": tag, "group": group, "tag": tag,
                             "members": names, "tta": EVAL_TTA, "best_iteration": None, "best_score3": s3,
                             "best_score4": s4, "best_pairs": pairs, "final_score3": s3, "final_score4": s4,
                             "shift": {}, "config": {}}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["pixdiff", "featdiff"], default=None)
    ap.add_argument("--no-mix", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--size", type=int, default=None, help="crop size of the cache (default 64)")
    ap.add_argument("--cache-suffix", default="", help="cache folder suffix: 'y' -> train_96y (fast luma decoding)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--validation-interval", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--batch-drivers", type=int, default=None)
    ap.add_argument("--batch-sources", type=int, default=None)
    ap.add_argument("--cross-samples", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--window-stride", type=int, default=None, help="evaluation window stride (frames)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--ema-decay", type=float, default=None, help="0 to disable")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-register", action="store_true", help="do not upload the best model to MLflow at the end")
    ap.add_argument("--input-size", type=int, default=None, help="network input size (e.g. 64 from a 96px cache)")
    ap.add_argument("--augment", action="store_true", help="renderer-style augmentation (see augment_batch)")
    ap.add_argument("--speed-min", type=float, default=None)
    ap.add_argument("--speed-max", type=float, default=None)
    ap.add_argument("--mixstyle", type=float, default=None, help="probability of MixStyle per training step")
    ap.add_argument("--embed-bn", action="store_true", help="BatchNorm on the embedding (training-set statistics)")
    ap.add_argument("--uniformity", type=float, default=None, help="weight of the uniformity loss")
    ap.add_argument("--run-tag", default=None, help="suffix for the MLflow run name (ablations)")
    ap.add_argument("--group", default=None, help="ablation-study group name (MLflow tag, study log)")
    ap.add_argument("--aug-parts", default=None, help="comma list from: amp,photo,blur,spatial,noise,speed")
    ap.add_argument("--aug-strength", type=float, default=None, help="scales all augmentations (1.0 = full)")
    ap.add_argument("--diff-lags", default=None, help="pixdiff difference lags, e.g. 1,2,4")
    ap.add_argument("--temporal-pool", choices=["avg", "attn"], default=None)
    ap.add_argument("--no-shift-eval", action="store_true", help="skip the synthetic-shift robustness proxy")
    ap.add_argument("--shift-eval-only", action="store_true", help="only score --checkpoint on clean + shifted validation")
    ap.add_argument("--tta-mode", choices=["geom", "render", "both"], default="geom",
                    help="test-time view family: geometric zoom/shift, renderer-like, or both")
    ap.add_argument("--eval-frames", type=int, default=0, help="evaluation window length in frames (0 = training value; 100 = whole clip)")
    ap.add_argument("--tta", type=int, default=1, help="test-time views per window for --test / --shift-eval-only (1-6)")
    ap.add_argument("--ensemble-eval", nargs="+", type=Path, default=None,
                    help="checkpoints whose cosine similarities are averaged and scored on the validation set")
    ap.add_argument("--width", type=float, default=None, help="channel multiplier of the network (capacity study)")
    ap.add_argument("--deflicker", action="store_true", help="per-frame intensity normalisation to clip statistics")
    ap.add_argument("--consistency", type=float, default=None,
                    help="weight of the clean-vs-perturbed embedding agreement term (0 = off)")
    ap.add_argument("--consistency-parts", default=None, help="rendering-only perturbations for the second view")
    ap.add_argument("--consistency-strength", type=float, default=None)
    ap.add_argument("--aux-landmark", type=float, default=None,
                    help="weight of the auxiliary landmark-embedding prediction loss (0 = off)")
    ap.add_argument("--arch", choices=["2d", "3d"], default=None, help="2d = per-frame conv + late 3D head (default); 3d = early spatio-temporal convs")
    ap.add_argument("--grad-checkpoint", action="store_true", help="activation checkpointing (fits larger batches)")
    ap.add_argument("--grad-cache", type=int, default=None, help="GradCache chunk size in clips (large batches, constant memory)")
    ap.add_argument("--mode-balance", action="store_true", help="one speaking + one listening clip per driver-source pair")
    ap.add_argument("--cross-mode-weight", type=float, default=None)
    ap.add_argument("--temporal-branch", action="store_true")
    ap.add_argument("--schedule-iterations", type=int, default=None)
    ap.add_argument("--io-threads", type=int, default=None)
    ap.add_argument("--test", action="store_true", help="evaluate --checkpoint on the validation set")
    ap.add_argument("--checkpoint", type=Path, default=None)
    a = ap.parse_args()

    base = Settings()
    o: dict[str, Any] = {}
    for cli, field in (("variant", "variant"), ("iterations", "iterations"),
                       ("validation_interval", "validation_interval"), ("lr", "learning_rate"),
                       ("weight_decay", "weight_decay"), ("temperature", "temperature"),
                       ("batch_drivers", "batch_drivers"), ("batch_sources", "batch_sources"),
                       ("cross_samples", "cross_samples"), ("frames", "frames"), ("window_stride", "window_stride"), ("workers", "workers"),
                       ("patience", "patience"), ("seed", "seed"),
                       ("warmup_steps", "warmup_steps"), ("ema_decay", "ema_decay"),
                       ("input_size", "input_size"), ("speed_min", "speed_min"), ("speed_max", "speed_max"),
                       ("mixstyle", "mixstyle"), ("uniformity", "uniformity"), ("run_tag", "run_tag"),
                       ("group", "group"), ("aug_parts", "aug_parts"), ("aug_strength", "aug_strength"),
                       ("diff_lags", "diff_lags"), ("temporal_pool", "temporal_pool"), ("width", "width"),
                       ("grad_cache", "grad_cache"), ("arch", "arch"), ("aux_landmark", "aux_landmark"), ("consistency", "consistency"),
                       ("consistency_parts", "consistency_parts"), ("consistency_strength", "consistency_strength"),
                       ("cross_mode_weight", "cross_mode_weight"), ("schedule_iterations", "schedule_iterations"),
                       ("io_threads", "io_threads")):
        if getattr(a, cli) is not None:
            o[field] = getattr(a, cli)
    if a.deflicker:
        o["deflicker"] = True
    if a.mode_balance:
        o["mode_balance"] = True
    if a.temporal_branch:
        o["temporal_branch"] = True
    if a.grad_checkpoint:
        o["grad_checkpoint"] = True
    if a.no_mix:
        o["use_mix"] = False
    if a.no_graph:
        o["use_graph"] = False
    if a.no_register:
        o["register"] = False
    if a.augment:
        o["augment"] = True
    if a.embed_bn:
        o["embed_bn"] = True
    if a.no_shift_eval:
        o["shift_eval"] = False
    if a.size is not None or a.cache_suffix:
        size = a.size if a.size is not None else base.image_size
        o["image_size"] = size
        o["cache_root"] = PROJECT_ROOT / f"data/video_crops/train_{size}{a.cache_suffix}"
    variant = o.get("variant", base.variant)
    o["output_dir"] = a.output_dir or PROJECT_ROOT / f"output/video/mdnet_{variant}" \
        f"{'_nomix' if a.no_mix else ''}{'_nograph' if a.no_graph else ''}"
    cfg = replace(base, **o)
    globals()["EVAL_TTA"] = a.tta
    globals()["EVAL_TTA_MODE"] = a.tta_mode
    globals()["EVAL_FRAMES"] = a.eval_frames
    if a.ensemble_eval:
        ensemble_only(a.ensemble_eval, a.group or "ensemble", a.run_tag or "ensemble")
    elif a.shift_eval_only:
        shift_only(a.checkpoint, a.group or "shift-reference", a.run_tag)
    elif a.test:
        test(cfg, a.checkpoint or cfg.output_dir / "best_model.pt")
    else:
        train(cfg)


if __name__ == "__main__":
    main()
