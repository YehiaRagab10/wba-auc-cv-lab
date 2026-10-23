#!/usr/bin/env python3
"""Kinematic-TDNN trainer for WBA2026 facial-landmark fingerprinting.

ECAPA-style 1D-TDNN on a compact kinematic feature front-end (303 channels)
or raw flattened landmarks (636 channels). Attentive statistics pooling
produces fixed-length embeddings for any input length.

Loss: L = L_AAM + 0.5·L_contrast + λ_adv·L_srcGRL

Examples:
    # Train with 303-channel kinematic front-end:
    python trainer.py --features kinematic --pca-path pca_basis.pt

    # Smoke test with raw features:
    python trainer.py --features raw

    # Evaluate a checkpoint:
    python trainer.py --test --checkpoint output/.../best_model.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import blosc2
import mlflow
import numpy as np
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from geometry import (
    KinematicFrontEnd,
    LandmarkAugmentation,
    RawFrontEnd,
)
from losses import KTDNNLoss

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

import sys
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from tracking import CSVMetricsLogger, setup_mlflow_run


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    data_root: Path = PROJECT_ROOT / "data/landmark_2d_106/train"
    meta_path: Path = PROJECT_ROOT / "data/meta/train-landmark_2d_106-meta.json"
    train_driver_split: Path = PROJECT_ROOT / "data/train_val_splits/driver_train_splits.json"
    train_source_split: Path = PROJECT_ROOT / "data/train_val_splits/source_train_splits.json"
    val_driver_split: Path = PROJECT_ROOT / "data/train_val_splits/driver_val_splits.json"
    val_source_split: Path = PROJECT_ROOT / "data/train_val_splits/source_val_splits.json"
    output_dir: Path = PROJECT_ROOT / "output/landmark_2d_106/ktdnn"

    # Feature mode: "kinematic" (303 channels) or "raw" (636 channels)
    features: str = "kinematic"
    pca_path: Path | None = None   # Path to PCA basis (required for kinematic)
    pca_k: int = 32

    # Architecture
    frames: int = 75               # clip length for training crops
    min_crop: int = 48             # variable-length: random crop T in [min_crop, max_crop]
    max_crop: int = 160
    vertices: int = 106
    channels: int = 303            # auto-set based on features mode
    tdnn_channels: int = 192 #384       # C in the ECAPA blocks
    embedding_dim: int = 256
    dropout_in: float = 0.1            # Dropout1d on input channels
    dropout_head: float = 0.3          # Dropout before final linear

    # Training
    batch_drivers: int = 32
    batch_sources: int = 4
    cross_samples: int = 1
    iterations: int = 10_000
    validation_interval: int = 500
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    grad_clip: float = 5.0
    warmup_steps: int = 500
    min_lr: float = 1e-5
    patience: int = 10
    ema_decay: float = 0.999
    temperature: float = 0.5
    lambda_contrast: float = 0.5
    lambda_adv_max: float = 0.1
    aam_s: float = 30.0
    aam_m: float = 0.2
    aam_k: int = 3
    margin_warmup_frac: float = 0.1   # full margin by ~step 800
    workers: int = 0
    seed: int = 0
    amp: bool = True
    eval_distance: str = "cosine"
    window_stride: int = 37        # for compatibility; not used (whole-sequence eval)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
        force=True,
    )


LOGGER = logging.getLogger("ktdnn-trainer")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return distributed.get_rank(), distributed.get_world_size(), local_rank


def is_main_process() -> bool:
    return not distributed.is_initialized() or distributed.get_rank() == 0


# ---------------------------------------------------------------------------
# Transforms (minimal — augmentations are GPU-based in LandmarkAugmentation)
# ---------------------------------------------------------------------------

class PadShortClips:
    """Pad short sequences by repeating the last frame."""

    def __init__(self, min_clip_length: int) -> None:
        self.min_clip_length = min_clip_length

    def __call__(self, data: Tensor) -> Tensor:
        if data.shape[0] < self.min_clip_length:
            pad_size = self.min_clip_length - data.shape[0]
            last_frame = data[-1:].expand(pad_size, *data.shape[1:])
            return torch.cat([data, last_frame], dim=0)
        return data


class RandomCrop:
    """Random temporal crop to a fixed length."""

    def __init__(self, clip_length: int) -> None:
        self.clip_length = clip_length

    def __call__(self, data: Tensor) -> Tensor:
        T = data.shape[0]
        if T <= self.clip_length:
            return data
        start = random.randint(0, T - self.clip_length)
        return data[start : start + self.clip_length]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrainLandmarkDataset(Dataset):
    """Blosc2 landmark dataset with separate driver/source ID mappings."""

    def __init__(
        self,
        root: Path,
        meta_path: Path,
        driver_split: Path,
        source_split: Path,
        min_frames: int,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"Landmark root does not exist: {root}")
        driver_ids = set(json.loads(driver_split.read_text()))
        source_ids = set(json.loads(source_split.read_text()))
        metadata = json.loads(meta_path.read_text())
        self.root = root
        self.transform = transform
        self.meta = [
            item.copy()
            for item in metadata
            if item["driver_id"] in driver_ids
            and item["source_id"] in source_ids
            and (root / item["relpath"]).is_file()
        ]
        if not self.meta:
            raise RuntimeError(f"No usable samples after filtering {meta_path}.")

        # Separate contiguous mappings for drivers and sources
        self.driver_to_class: dict[str, int] = {}
        self.source_to_class: dict[str, int] = {}
        for item in self.meta:
            did = item["driver_id"]
            sid = item["source_id"]
            if did not in self.driver_to_class:
                self.driver_to_class[did] = len(self.driver_to_class)
            if sid not in self.source_to_class:
                self.source_to_class[sid] = len(self.source_to_class)
            item["driver_class"] = self.driver_to_class[did]
            item["source_class"] = self.source_to_class[sid]

        self.num_drivers = len(self.driver_to_class)
        self.num_sources = len(self.source_to_class)

        LOGGER.info(
            "Loaded %d samples, %d drivers, %d sources from %s",
            len(self.meta), self.num_drivers, self.num_sources, meta_path,
        )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.meta[index]
        landmarks = blosc2.load_tensor(self.root / item["relpath"])
        if not isinstance(landmarks, Tensor):
            landmarks = torch.as_tensor(landmarks)
        if self.transform is not None:
            landmarks = self.transform(landmarks)
        return {
            "landmarks_xyz": landmarks,
            "driver_class": item["driver_class"],
            "source_class": item["source_class"],
            "driving_id": item["driver_class"],  # alias for evaluate()
            "source_id": item["source_class"],
            "mode": item.get("mode", "speaking"),
            "set": item.get("set", "gallery"),
            "file": (self.root / item["relpath"]).as_posix(),
        }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class DriverSourceSampler(Sampler[list[int]]):
    """Samples p drivers × ps sources × both modes for contrastive batches.

    Includes self-reenactments when available.
    """

    def __init__(
        self,
        dataset: TrainLandmarkDataset,
        p: int,
        ps: int,
        k_cross: int,
        iterations: int,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        self.p, self.ps, self.k_cross, self.iterations = p, ps, k_cross, iterations
        self.rank, self.world_size = rank, world_size
        self.rng = np.random.default_rng(seed + rank)
        self.by_dsm: dict[tuple[int, int, str], list[int]] = defaultdict(list)
        self.by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.meta):
            d, s = item["driver_class"], item["source_class"]
            mode = item.get("mode", "speaking")
            self.by_dsm[d, s, mode].append(index)
            self.by_pair[d, s].append(index)
        self.drivers = sorted({d for d, _ in self.by_pair})
        self.sources = sorted({s for _, s in self.by_pair})
        self.modes = ("speaking", "listening")
        if not self.drivers or not self.sources:
            raise RuntimeError("The training split has no usable samples.")

    def _choice(self, values: Sequence[int], count: int) -> list[int]:
        if not values:
            return []
        return self.rng.choice(values, size=count, replace=len(values) < count).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.iterations):
            drivers = self._choice(self.drivers, self.p)
            compatible_sources = [
                s for s in self.sources if all((d, s) in self.by_pair for d in drivers)
            ]
            if len(compatible_sources) >= self.ps:
                sources = self._choice(compatible_sources, self.ps)
            else:
                sources = self._choice(self.sources, self.ps)

            flat: list[int] = []
            for d in drivers:
                for s in sources:
                    for mode in self.modes:
                        bucket = self.by_dsm.get((d, s, mode), [])
                        if bucket:
                            flat.extend(self._choice(bucket, self.k_cross))

            usable = len(flat) - (len(flat) % max(1, self.world_size))
            if usable == 0:
                continue
            yield flat[:usable][self.rank : usable : self.world_size]

    def __len__(self) -> int:
        return self.iterations


# ---------------------------------------------------------------------------
# ECAPA-TDNN model components
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        w = self.se(x).unsqueeze(-1)  # [B, C, 1]
        return x * w


class Res2Conv1d(nn.Module):
    """Res2Net-style multi-scale 1D convolution.

    Splits input channels into ``scale`` groups, processes them with
    cascading convolutions to capture multi-scale context.
    """

    def __init__(
        self, channels: int, kernel_size: int, dilation: int, scale: int = 8
    ) -> None:
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width, self.width, kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation, bias=False,
            )
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        chunks = x.chunk(self.scale, dim=1)
        outs = [chunks[0]]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0:
                y = conv(chunks[i + 1])
            else:
                y = conv(chunks[i + 1] + outs[-1])
            outs.append(F.relu(bn(y), inplace=True))
        return torch.cat(outs, dim=1)


class SERes2Block(nn.Module):
    """SE-Res2Block: Conv1d → BN → ReLU → Res2Conv1d → BN → ReLU → Conv1d → BN → SE + residual."""

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: int = 2, scale: int = 8
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            Res2Conv1d(channels, kernel_size, dilation, scale),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.se(self.block(x)))


class AttentiveStatisticsPooling(nn.Module):
    """Channel- and context-dependent attentive statistics pooling.

    Produces [B, 2C] output: concatenation of weighted mean and weighted std.
    Supports an optional mask for variable-length inputs.
    """

    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attention_channels, channels, 1),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, C, T]
            mask: [B, T] boolean, True for valid frames

        Returns:
            [B, 2C]
        """
        # Attention weights
        alpha = self.attention(x)  # [B, C, T]
        if mask is not None:
            alpha = alpha.masked_fill(~mask.unsqueeze(1), -torch.inf)
        alpha = F.softmax(alpha, dim=-1)  # [B, C, T]

        # Weighted mean
        mu = (alpha * x).sum(dim=-1)  # [B, C]

        # Weighted std
        var = (alpha * (x - mu.unsqueeze(-1)).square()).sum(dim=-1).clamp_min(1e-8)
        sigma = var.sqrt()  # [B, C]

        return torch.cat([mu, sigma], dim=1)  # [B, 2C]


# ---------------------------------------------------------------------------
# ECAPA-TDNN Model
# ---------------------------------------------------------------------------

class KinematicTDNN(nn.Module):
    """ECAPA-style TDNN for kinematic landmark features.

    Architecture (≈3-5M params):
        [B, in_channels, T]
        → Conv1d(in_channels, C=384, k=5) + BN + ReLU
        → SE-Res2Block(C, k=3, dilation=2, scale=8)
        → SE-Res2Block(C, k=3, dilation=3, scale=8)
        → SE-Res2Block(C, k=3, dilation=4, scale=8)
        → concat 3 block outputs → Conv1d(3C, 3C, 1) + ReLU
        → AttentiveStatisticsPooling → [B, 6C]  (μ ∥ σ)
        → BN → Linear(6C, 256) → BN  = embedding
    """

    def __init__(
        self,
        in_channels: int = 303,
        tdnn_channels: int = 384,
        embedding_dim: int = 256,
        dropout_in: float = 0.1,
        dropout_head: float = 0.3,
    ) -> None:
        super().__init__()
        C = tdnn_channels

        self.input_bn = nn.BatchNorm1d(in_channels)
        self.in_drop = nn.Dropout1d(dropout_in)

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )

        self.block1 = SERes2Block(C, kernel_size=3, dilation=2, scale=8)
        self.block2 = SERes2Block(C, kernel_size=3, dilation=3, scale=8)
        self.block3 = SERes2Block(C, kernel_size=3, dilation=4, scale=8)

        # Multi-layer aggregation
        self.mla = nn.Sequential(
            nn.Conv1d(3 * C, 3 * C, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.asp = AttentiveStatisticsPooling(3 * C)

        self.head = nn.Sequential(
            nn.BatchNorm1d(6 * C),
            nn.Dropout(dropout_head),
            nn.Linear(6 * C, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            x: [B, in_channels, T]
            mask: [B, T] boolean, True for valid frames (optional)

        Returns:
            embedding: [B, embedding_dim]  (NOT L2-normalised — loss does that)
        """
        x = self.input_bn(x)
        x = self.in_drop(x)
        x = self.stem(x)

        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        # Multi-layer aggregation
        cat = torch.cat([out1, out2, out3], dim=1)  # [B, 3C, T]
        cat = self.mla(cat)

        # Attentive statistics pooling
        pooled = self.asp(cat, mask)  # [B, 6C]

        # Embedding head
        embedding = self.head(pooled)  # [B, embedding_dim]
        return embedding


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = model.module if isinstance(model, DDP) else model
        for ema_p, model_p in zip(self.module.parameters(), src.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
        for ema_b, model_b in zip(self.module.buffers(), src.buffers()):
            ema_b.data.copy_(model_b.data)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.LambdaLR:
    base_lr = optimizer.defaults["lr"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_auc(scores: Tensor, targets: Tensor) -> float:
    positives = int(targets.sum())
    negatives = int((1 - targets).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    targets = targets[order]
    tpr = targets.cumsum(0).float() / positives
    fpr = (1 - targets).cumsum(0).float() / negatives
    return torch.trapz(
        torch.cat((torch.zeros(1, device=tpr.device), tpr)),
        torch.cat((torch.zeros(1, device=fpr.device), fpr)),
    ).item()


def compute_eer(scores: Tensor, targets: Tensor) -> float:
    positives = int(targets.sum())
    negatives = int((1 - targets).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    sorted_targets = targets[order]
    tpr = sorted_targets.cumsum(0).float() / positives
    fpr = (1 - sorted_targets).cumsum(0).float() / negatives
    fnr = 1.0 - tpr
    diff = fpr - fnr
    idx = (diff[:-1] * diff[1:] <= 0).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return (fpr[-1].item() + fnr[-1].item()) / 2
    i = idx[0].item()
    d0, d1 = diff[i].item(), diff[i + 1].item()
    alpha = d0 / (d0 - d1) if d0 != d1 else 0.5
    return fpr[i].item() * (1 - alpha) + fpr[i + 1].item() * alpha


def _compute_pair_metrics(
    probe_emb: Tensor,
    probe_ids: Tensor,
    gallery_emb: Tensor,
    gallery_ids: Tensor,
    distance_metric: str = "cosine",
) -> tuple[float, float]:
    if distance_metric == "euclidean":
        distances = torch.cdist(probe_emb, gallery_emb, p=2)
    elif distance_metric == "cosine":
        p_n = F.normalize(probe_emb, p=2, dim=-1)
        g_n = F.normalize(gallery_emb, p=2, dim=-1)
        distances = 1.0 - (p_n @ g_n.T)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    targets = (probe_ids.unsqueeze(1) == gallery_ids.unsqueeze(0)).flatten().float()
    similarities = -distances.flatten()
    return compute_auc(similarities, targets), compute_eer(similarities, targets)


def compute_official_score(metrics: dict[str, float]) -> float:
    """Official WBA2026 score: mean of 3 conditions."""
    speaking = metrics.get("emb_speaking_vs_speaking_auc", float("nan"))
    listening = metrics.get("emb_listening_vs_listening_auc", float("nan"))
    cross_sl = metrics.get("emb_speaking_vs_listening_auc", float("nan"))
    cross_ls = metrics.get("emb_listening_vs_speaking_auc", float("nan"))
    cross = (
        (cross_sl + cross_ls) / 2
        if not (math.isnan(cross_sl) or math.isnan(cross_ls))
        else float("nan")
    )
    values = [v for v in [speaking, listening, cross] if not math.isnan(v)]
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Postprocessing (shared between evaluate and infer)
# ---------------------------------------------------------------------------

def postprocess(embeddings: Tensor) -> Tensor:
    """Post-process embeddings: L2-normalise and optionally center.

    Called by both evaluate() and infer.py to ensure consistency.
    """
    # Center embeddings (mean subtraction)
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    # L2-normalise
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(config: Settings, device: torch.device) -> nn.Module:
    """Build front-end + TDNN model."""
    if config.features == "kinematic":
        frontend = KinematicFrontEnd(pca_k=config.pca_k)
        if config.pca_path is not None and config.pca_path.exists():
            pca_data = torch.load(config.pca_path, map_location="cpu", weights_only=True)
            frontend.load_pca(pca_data["basis"][:config.pca_k], pca_data["mean"])
            LOGGER.info("Loaded PCA basis from %s", config.pca_path)
        in_channels = 303
    else:
        frontend = RawFrontEnd()
        in_channels = 636

    tdnn = KinematicTDNN(
        in_channels=in_channels,
        tdnn_channels=config.tdnn_channels,
        embedding_dim=config.embedding_dim,
        dropout_in=config.dropout_in,
        dropout_head=config.dropout_head,
    )

    model = KTDNNWithFrontEnd(frontend, tdnn)
    return model.to(device)


class KTDNNWithFrontEnd(nn.Module):
    """Wraps front-end + TDNN into a single module for checkpointing."""

    def __init__(self, frontend: nn.Module, tdnn: KinematicTDNN) -> None:
        super().__init__()
        self.frontend = frontend
        self.tdnn = tdnn

    def forward(
        self, landmarks: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            landmarks: [B, T, 106, 2] raw pixel coordinates
            mask: [B, T] boolean, True for valid frames

        Returns:
            embedding: [B, embedding_dim]
        """
        features = self.frontend(landmarks)  # [B, C, T]
        return self.tdnn(features, mask)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    iteration: int,
    metrics: dict[str, float],
    ema: ModelEMA | None = None,
    heads: dict[str, Any] | None = None,
) -> None:
    module = model.module if isinstance(model, DDP) else model
    state: dict[str, Any] = {
        "iteration": iteration,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    if ema is not None:
        state["ema"] = ema.module.state_dict()
    if heads is not None:
        state["heads"] = heads
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ModelEMA | None = None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    module = model.module if isinstance(model, DDP) else model
    module.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if ema is not None and checkpoint.get("ema"):
        ema.module.load_state_dict(checkpoint["ema"])
    return int(checkpoint.get("iteration", 0))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _varlen_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate that pads variable-length landmark tensors to the max
    length in the batch. Returns a ``lengths`` tensor alongside the padded data."""
    lengths = torch.tensor([b["landmarks_xyz"].shape[0] for b in batch], dtype=torch.long)
    max_t = int(lengths.max().item())

    padded = []
    for b in batch:
        lm = b["landmarks_xyz"]  # [T_i, 106, 2]
        if lm.shape[0] < max_t:
            pad_size = max_t - lm.shape[0]
            lm = torch.cat([lm, lm[-1:].expand(pad_size, *lm.shape[1:])], dim=0)
        padded.append(lm)

    result: dict[str, Any] = {
        "landmarks_xyz": torch.stack(padded, dim=0),  # [B, T_max, 106, 2]
        "lengths": lengths,                            # [B]
    }
    # Collate scalar / string fields normally
    for key in ("driver_class", "source_class", "driving_id", "source_id"):
        result[key] = torch.tensor([b[key] for b in batch])
    for key in ("mode", "set", "file"):
        result[key] = [b[key] for b in batch]
    return result


def build_loaders(
    config: Settings, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader]:
    # Training: pad short clips, random crop done per-batch on GPU
    train_transform = PadShortClips(config.min_crop)

    train_dataset = TrainLandmarkDataset(
        config.data_root,
        config.meta_path,
        config.train_driver_split,
        config.train_source_split,
        config.min_crop,
        train_transform,
    )

    # Validation: pad only, full sequence evaluation
    val_transform = PadShortClips(config.min_crop)

    val_dataset = TrainLandmarkDataset(
        config.data_root,
        config.meta_path,
        config.val_driver_split,
        config.val_source_split,
        config.min_crop,
        val_transform,
    )

    sampler = DriverSourceSampler(
        train_dataset,
        config.batch_drivers,
        config.batch_sources,
        config.cross_samples,
        config.iterations,
        config.seed,
        rank,
        world_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=config.workers,
        pin_memory=False,
        persistent_workers=config.workers > 0,
        worker_init_fn=_worker_init_fn,
        collate_fn=_varlen_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=config.workers,
        pin_memory=False,
        persistent_workers=config.workers > 0,
        worker_init_fn=_worker_init_fn,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation — whole-sequence, no sliding windows
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    distance_metric: str = "cosine",
    frames: int = 75,
    window_stride: int = 37,
) -> dict[str, float]:
    """Evaluate with whole-sequence forward passes (1D convs handle any length)."""
    model.eval()
    buckets: dict[tuple[str, str], dict[str, list[Tensor]]] = {
        ("probe", "speaking"): {"emb": [], "driving": []},
        ("probe", "listening"): {"emb": [], "driving": []},
        ("gallery", "speaking"): {"emb": [], "driving": []},
        ("gallery", "listening"): {"emb": [], "driving": []},
    }

    for batch in loader:
        raw = batch["landmarks_xyz"]  # [1, T, V, C] or [1, T, 106, 2]
        x = raw.to(device, non_blocking=True).float()

        # Whole-sequence forward
        embedding = model(x)  # [1, D]
        embedding = F.normalize(embedding, p=2, dim=-1)

        modes = batch["mode"]
        sets = batch["set"]
        driving = batch["driving_id"].cpu()

        for i in range(len(modes)):
            key = (sets[i], modes[i])
            if key in buckets:
                buckets[key]["emb"].append(embedding[i].cpu())
                buckets[key]["driving"].append(driving[i])

    for key in buckets:
        b = buckets[key]
        b["emb"] = torch.stack(b["emb"]) if b["emb"] else torch.empty(0, 256)
        b["driving"] = (
            torch.stack(b["driving"]).long() if b["driving"] else torch.empty(0).long()
        )

    # Post-process all embeddings together for centering
    all_embs = []
    all_keys = []
    for key in buckets:
        if buckets[key]["emb"].numel() > 0:
            all_embs.append(buckets[key]["emb"])
            all_keys.append(key)
    if all_embs:
        combined = torch.cat(all_embs, dim=0)
        combined = postprocess(combined)
        offset = 0
        for key in all_keys:
            n = buckets[key]["emb"].shape[0]
            buckets[key]["emb"] = combined[offset : offset + n]
            offset += n

    output: dict[str, float] = {}
    for probe_mode in ("speaking", "listening"):
        for gallery_mode in ("speaking", "listening"):
            p = buckets[("probe", probe_mode)]
            g = buckets[("gallery", gallery_mode)]
            if p["emb"].numel() == 0 or g["emb"].numel() == 0:
                continue
            auc, eer = _compute_pair_metrics(
                p["emb"], p["driving"], g["emb"], g["driving"], distance_metric
            )
            tag = f"emb_{probe_mode}_vs_{gallery_mode}"
            output[f"{tag}_auc"] = auc
            output[f"{tag}_eer"] = eer

    return output


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: Settings, resume: Path | None = None) -> None:
    rank, world_size, local_rank = distributed_context()
    setup_logging(config.output_dir)
    seed_everything(config.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config, rank, world_size)

    # Get num_drivers / num_sources from the training dataset
    train_dataset: TrainLandmarkDataset = train_loader.dataset  # type: ignore
    num_drivers = train_dataset.num_drivers
    num_sources = train_dataset.num_sources
    LOGGER.info("AAM classes: %d drivers, GRL classes: %d sources", num_drivers, num_sources)

    # Tracking
    csv_logger: CSVMetricsLogger | None = None
    if is_main_process():
        setup_mlflow_run("ktdnn", "ktdnn-train", config)
        csv_logger = CSVMetricsLogger(config.output_dir)

    raw_model = _build_model(config, device)
    model: nn.Module = raw_model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # Loss (heads are separate from the model)
    criterion = KTDNNLoss(
        embedding_dim=config.embedding_dim,
        num_drivers=num_drivers,
        num_sources=num_sources,
        s=config.aam_s,
        m=config.aam_m,
        k=config.aam_k,
        temperature=config.temperature,
        lambda_contrast=config.lambda_contrast,
        lambda_adv_max=config.lambda_adv_max,
    ).to(device)

    # Augmentation module (training only)
    augmenter = LandmarkAugmentation().to(device)

    # Optimizer: separate weight decay for conv/linear vs BN/bias
    decay_params = []
    no_decay_params = []
    for name, param in list(model.named_parameters()) + list(criterion.named_parameters()):
        if not param.requires_grad:
            continue
        # BN weight/bias are 1D; linear/conv bias is also 1D — no decay on any of them
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.iterations, config.min_lr
    )
    amp_enabled = config.amp and device.type == "cuda"

    # EMA
    ema: ModelEMA | None = None
    if config.ema_decay > 0:
        ema = ModelEMA(raw_model, config.ema_decay)

    start = 0
    if resume:
        start = load_checkpoint(resume, model, optimizer, scheduler, ema)
        # Load heads if present
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        if "heads" in ckpt:
            criterion.load_state_dict(ckpt["heads"])
            LOGGER.info("Loaded loss heads from checkpoint")

    best_score = -math.inf
    patience_counter = 0
    loader_iterator = iter(train_loader)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Model parameters: %d total, %d trainable", total_params, trainable)

    for iteration in range(start + 1, config.iterations + 1):
        model.train()
        criterion.train()
        batch = next(loader_iterator)

        # Raw landmarks: [B, T_max, 106, 2] (padded by collate)
        landmarks = batch["landmarks_xyz"].to(device, non_blocking=True).float()
        lengths = batch["lengths"].to(device)  # [B] actual lengths before padding

        driver_labels = batch["driver_class"].to(device, non_blocking=True).long()
        source_labels = batch["source_class"].to(device, non_blocking=True).long()

        # Variable-length random crop: same crop_len for the batch, per-sample start
        B = landmarks.shape[0]
        crop_len = random.randint(config.min_crop, config.max_crop)
        crop_len = min(crop_len, int(lengths.min().item()))
        max_start = (lengths - crop_len).clamp_min(0)
        starts = (torch.rand(B, device=device) * (max_start + 1).float()).long()
        idx = starts[:, None] + torch.arange(crop_len, device=device)  # [B, crop_len]
        landmarks = landmarks[torch.arange(B, device=device)[:, None], idx]  # [B, crop_len, 106, 2]

        # GPU augmentations on raw landmarks
        aug_landmarks = augmenter(landmarks)

        # Also create a time-shuffled copy (shuffle before front-end so Δ/ΔΔ reflect broken order)
        T_aug = aug_landmarks.shape[1]
        perm = torch.randperm(T_aug, device=device)
        shuffled_landmarks = aug_landmarks[:, perm]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16,
        ):
            # Forward original
            emb_original = model(aug_landmarks)  # [B, D]
            # Forward shuffled
            emb_shuffled = model(shuffled_landmarks)  # [B, D]

        # Margin warm-up: 0 → 1 over first 20% of steps
        margin_warmup_steps = int(config.iterations * config.margin_warmup_frac)
        margin_scale = min(1.0, iteration / max(1, margin_warmup_steps))

        # Adversarial ramp: 0 → 1 over first 30% of training
        adv_scale = min(1.0, iteration / (0.3 * config.iterations))

        # Loss in float32
        loss, info = criterion(
            emb_original.float(),
            emb_shuffled.float(),
            driver_labels,
            source_labels,
            margin_scale=margin_scale,
            adv_scale=adv_scale,
        )

        if not torch.isfinite(loss):
            LOGGER.warning("Non-finite loss at iteration %d — skipping", iteration)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        loss.backward()
        if config.grad_clip > 0:
            all_params = list(model.parameters()) + list(criterion.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, config.grad_clip)
        else:
            grad_norm = None
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        if is_main_process() and (iteration == 1 or iteration % 20 == 0):
            grad_str = f" | grad_norm {grad_norm:.4f}" if grad_norm is not None else ""
            LOGGER.info(
                "iter %d/%d | loss %.4f | aam %.4f | contrast %.4f | grl %.4f | lr %.2e%s",
                iteration, config.iterations,
                info["loss_total"], info["loss_aam"],
                info["loss_contrast"], info["loss_src_grl"],
                optimizer.param_groups[0]["lr"], grad_str,
            )
            train_metrics = {k: v for k, v in info.items()}
            train_metrics["lr"] = optimizer.param_groups[0]["lr"]
            train_metrics["margin_scale"] = margin_scale
            mlflow.log_metrics(
                {f"train/{k}": v for k, v in train_metrics.items()},
                step=iteration,
            )
            if csv_logger is not None:
                csv_logger.log("train", train_metrics, step=iteration)

        if (
            iteration % config.validation_interval == 0
            or iteration == config.iterations
        ):
            if distributed.is_initialized():
                distributed.barrier()

            early_stop = False
            if is_main_process():
                eval_model = model.module if isinstance(model, DDP) else model
                metrics = evaluate(
                    eval_model, val_loader, device,
                    distance_metric=config.eval_distance,
                    frames=config.frames,
                    window_stride=config.window_stride,
                )
                live_score = compute_official_score(metrics)
                metrics["score"] = live_score

                if ema is not None:
                    ema_metrics = evaluate(
                        ema.module, val_loader, device,
                        distance_metric=config.eval_distance,
                        frames=config.frames,
                        window_stride=config.window_stride,
                    )
                    ema_score = compute_official_score(ema_metrics)
                    metrics.update({f"ema_{k}": v for k, v in ema_metrics.items()})
                    metrics["ema_score"] = ema_score
                    current_score = max(ema_score, live_score)
                    score_source = "EMA" if ema_score >= live_score else "live"
                else:
                    current_score = live_score
                    score_source = "live"

                LOGGER.info(
                    "val at %d | score %.6f (%s) | %s",
                    iteration, current_score, score_source,
                    ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
                )
                mlflow.log_metrics(
                    {f"val/{k}": v for k, v in metrics.items()},
                    step=iteration,
                )
                if csv_logger is not None:
                    csv_logger.log("val", metrics, step=iteration)

                heads_state = criterion.state_dict()
                save_checkpoint(
                    config.output_dir / "last_model.pt",
                    model, optimizer, scheduler, iteration, metrics, ema,
                    heads=heads_state,
                )

                if current_score > best_score:
                    best_score = current_score
                    patience_counter = 0
                    if ema is not None and score_source == "EMA":
                        torch.save(
                            {
                                "iteration": iteration,
                                "model": ema.module.state_dict(),
                                "optimizer": None,
                                "scheduler": None,
                                "metrics": metrics,
                                "heads": heads_state,
                            },
                            config.output_dir / "best_model.pt",
                        )
                    else:
                        save_checkpoint(
                            config.output_dir / "best_model.pt",
                            model, optimizer, scheduler, iteration, metrics, ema,
                            heads=heads_state,
                        )
                    LOGGER.info(
                        "New best (%s) at iter %d: score %.6f",
                        score_source, iteration, best_score,
                    )
                    mlflow.log_metric("val/best_score", best_score, step=iteration)
                else:
                    patience_counter += 1
                    if config.patience > 0 and patience_counter >= config.patience:
                        LOGGER.info(
                            "Early stopping at %d (no improvement for %d validations)",
                            iteration, config.patience,
                        )
                        early_stop = True

            if distributed.is_initialized():
                stop_t = torch.tensor(
                    [int(early_stop)], device=device, dtype=torch.long
                )
                distributed.broadcast(stop_t, 0)
                early_stop = stop_t.item() == 1
                distributed.barrier()

            if early_stop:
                break

    if is_main_process():
        if csv_logger is not None:
            csv_logger.close()
        mlflow.end_run()

    if distributed.is_initialized():
        distributed.destroy_process_group()


def test(config: Settings, checkpoint: Path) -> None:
    _, _, local_rank = distributed_context()
    setup_logging(config.output_dir)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_loaders(config, 0, 1)
    model = _build_model(config, device)
    load_checkpoint(checkpoint, model)
    metrics = evaluate(
        model, val_loader, device,
        distance_metric=config.eval_distance,
        frames=config.frames,
        window_stride=config.window_stride,
    )
    score = compute_official_score(metrics)
    LOGGER.info(
        "test | score=%.6f | %s",
        score,
        ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
    )
    if distributed.is_initialized():
        distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CONFIG = Settings()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=CONFIG.output_dir / "best_model.pt",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--features", type=str, choices=["kinematic", "raw"], default=None,
    )
    parser.add_argument("--pca-path", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-drivers", type=int, default=None)
    parser.add_argument("--batch-sources", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--tdnn-channels", type=int, default=None)
    parser.add_argument("--pca-k", type=int, default=None)
    parser.add_argument("--lambda-contrast", type=float, default=None)
    parser.add_argument("--lambda-adv-max", type=float, default=None)
    parser.add_argument("--dropout-in", type=float, default=None)
    parser.add_argument("--dropout-head", type=float, default=None)
    parser.add_argument(
        "--eval-distance", type=str, choices=["cosine", "euclidean"], default=None,
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    _cli_map = {
        "lr": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_drivers": "batch_drivers",
        "batch_sources": "batch_sources",
        "ema_decay": "ema_decay",
        "warmup_steps": "warmup_steps",
        "eval_distance": "eval_distance",
        "tdnn_channels": "tdnn_channels",
        "pca_k": "pca_k",
        "lambda_contrast": "lambda_contrast",
        "lambda_adv_max": "lambda_adv_max",
        "dropout_in": "dropout_in",
        "dropout_head": "dropout_head",
    }
    for cli_attr, field_name in _cli_map.items():
        val = getattr(args, cli_attr, None)
        if val is not None:
            overrides[field_name] = val
    for name in ("iterations", "seed", "patience", "features", "pca_path"):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val

    # Update channels based on features mode
    features_mode = overrides.get("features", CONFIG.features)
    if features_mode == "raw":
        overrides["channels"] = 636
    else:
        overrides["channels"] = 303

    config = replace(CONFIG, **overrides) if overrides else CONFIG

    if args.test:
        test(config, args.checkpoint)
    else:
        train(config, args.checkpoint if args.resume else None)


if __name__ == "__main__":
    main()