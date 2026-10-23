#!/usr/bin/env python3
"""NVIDIA-style trainer for WBA2026 — Temporal CNN on pairwise landmark distances.

Based on "Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos"
(Prashnani et al., NVIDIA).

Key design choices (from the paper):
  - Features: IED-normalized pairwise landmark distances (5565-dim per frame).
    These are inherently translation/rotation invariant and capture facial geometry
    changes over time — the "dynamic identity signature."
  - Architecture: Temporal ResNet (Conv1d) processes distance trajectories across
    71-frame clips, outputting L2-normalized identity embeddings.
  - Loss: 3-term contrastive — pull (same driver), push (same source / different
    driver), time-shuffle push (temporally shuffled same-driver clips treated as
    negatives to enforce temporal dynamics learning).
  - Sampling: 8 driving identities per batch with multiple sources per driver,
    including self-reenactments when available.

Examples:
    python trainer.py
    python trainer.py --test --checkpoint best_model.pt
    python trainer.py --batch-drivers 8 --lr 3e-4
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

import sys
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent))
from tracking import CSVMetricsLogger, setup_mlflow_run


# ---------------------------------------------------------------------------
# Semantic landmark regions (InsightFace 2D-106)
# ---------------------------------------------------------------------------

LANDMARK_REGIONS = {
    "face_contour": list(range(0, 33)),
    "left_eye": list(range(33, 43)),
    "left_eyebrow": list(range(43, 52)),
    "mouth": list(range(52, 72)),
    "nose": list(range(72, 87)),
    "right_eye": list(range(87, 97)),
    "right_eyebrow": list(range(97, 106)),
}


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
    output_dir: Path = PROJECT_ROOT / "output/landmark_2d_106/nvidia_tcnn"

    frames: int = 71               # NVIDIA uses 71-frame clips
    vertices: int = 106
    embedding_dim: int = 256
    stage_channels: tuple[int, ...] = (256, 256, 512)
    stage_blocks: tuple[int, ...] = (2, 2, 2)
    stage_kernels: tuple[int, ...] = (7, 5, 3)

    batch_drivers: int = 8         # NVIDIA: 8 identities per batch
    batch_sources: int = 4
    cross_samples: int = 1         # per mode — cross-mode sampling gives 2× this
    iterations: int = 30_000
    validation_interval: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    drop_path: float = 0.1
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    patience: int = 0              # 0 = disabled; let full cosine schedule play out
    ema_decay: float = 0.999
    window_stride: int = 35
    embed_bn: bool = False         # BatchNorm on the embedding (training statistics, frozen at eval)
    temperature: float = 0.5
    workers: int = 2
    seed: int = 0
    amp: bool = True
    eval_distance: str = "cosine"
    # "two_stream": static distances ∥ per-clip-demeaned distances (original NVIDIA recipe).
    # "three_stream_velocity": adds a third stream of frame-to-frame distance velocity,
    # on the hypothesis that it's more renderer-invariant than absolute geometry.
    feature_mode: str = "two_stream"
    # "cnn": original temporal ResNet (PairwiseDistanceCNN).
    # "transformer": self-attention backbone (PairwiseDistanceTransformer), same features/loss.
    arch: str = "cnn"

    @property
    def num_pairs(self) -> int:
        return self.vertices * (self.vertices - 1) // 2  # 5565

    @property
    def feature_streams(self) -> int:
        return 3 if self.feature_mode == "three_stream_velocity" else 2


CONFIG = Settings()

LOGGER = logging.getLogger("trainer")


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
# Transforms
# ---------------------------------------------------------------------------

class Compose:
    def __init__(self, transforms: Sequence[Callable[[Tensor], Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, data: Tensor) -> Tensor:
        for transform in self.transforms:
            data = transform(data)
        return data


class PadShortClips:
    """Pad short sequences by reflecting frames to preserve motion dynamics."""

    def __init__(self, min_clip_length: int) -> None:
        self.min_clip_length = min_clip_length

    def __call__(self, data: Tensor) -> Tensor:
        T = data.shape[0]
        if T >= self.min_clip_length:
            return data
        if T == 1:
            return data.expand(self.min_clip_length, *data.shape[1:])
        # Reflect: [0,1,2,3] → [0,1,2,3,2,1,0,1,2,3,...]
        cycle = list(range(T)) + list(range(T - 2, 0, -1))
        indices: list[int] = []
        while len(indices) < self.min_clip_length:
            indices.extend(cycle)
        return data[torch.tensor(indices[: self.min_clip_length])]


class RandomClipWithDrop:
    def __init__(self, clip_length: int, drop_rate: float) -> None:
        self.clip_length = clip_length
        self.drop_rate = drop_rate

    def __call__(self, data: Tensor) -> Tensor:
        if data.shape[0] == self.clip_length:
            return data
        if data.shape[0] < self.clip_length:
            raise ValueError(
                f"Expected at least {self.clip_length} frames, got {data.shape[0]}."
            )
        dynamic_length = min(
            int(self.clip_length * (1 + self.drop_rate)), data.shape[0]
        )
        start = torch.randint(data.shape[0] - dynamic_length + 1, ()).item()
        indices = (
            torch.randperm(dynamic_length)[: self.clip_length].sort().values + start
        )
        return data[indices]


class FaceRelativeNormalize:
    """Center on face centroid, align eyes horizontally, scale by inter-eye distance.

    Even though pairwise distances are translation/rotation invariant, this
    normalization is applied BEFORE SpatialAugment so that augmentation operates
    on a canonical face pose, making noise/scale/rotation perturbations consistent.
    """

    LEFT_EYE = list(range(33, 43))
    RIGHT_EYE = list(range(87, 97))

    def __call__(self, data: Tensor) -> Tensor:
        data = data.float()
        centroid = data.mean(dim=-2, keepdim=True)
        centered = data - centroid
        left_center = centered[:, self.LEFT_EYE, :].mean(dim=1)
        right_center = centered[:, self.RIGHT_EYE, :].mean(dim=1)
        diff = right_center - left_center
        angle = torch.atan2(diff[:, 1], diff[:, 0])
        cos_a = torch.cos(-angle).unsqueeze(1).unsqueeze(2)
        sin_a = torch.sin(-angle).unsqueeze(1).unsqueeze(2)
        x = centered[..., 0:1]
        y = centered[..., 1:2]
        rotated = torch.cat(
            [x * cos_a - y * sin_a, x * sin_a + y * cos_a], dim=-1
        )
        left_new = rotated[:, self.LEFT_EYE, :].mean(dim=1, keepdim=True)
        right_new = rotated[:, self.RIGHT_EYE, :].mean(dim=1, keepdim=True)
        ied = (left_new - right_new).norm(dim=-1, keepdim=True).clamp_min(1e-6)
        result = rotated / ied
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class SpatialAugment:
    """Random rotation, scale, and Gaussian noise on (x, y) coordinates."""

    def __init__(
        self,
        noise_std: float = 0.02,
        scale_range: tuple[float, float] = (0.9, 1.1),
        rotate_deg: float = 15.0,
    ) -> None:
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.rotate_deg = rotate_deg

    def __call__(self, data: Tensor) -> Tensor:
        angle = random.uniform(-self.rotate_deg, self.rotate_deg)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = torch.tensor(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=data.dtype, device=data.device
        )
        data = data @ rot.T
        scale = random.uniform(*self.scale_range)
        data = data * scale
        if self.noise_std > 0:
            data = data + torch.randn_like(data) * self.noise_std
        return data


class TemporalSpeedPerturb:
    """Randomly resample frames at perturbed speed."""

    def __init__(self, speed_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.speed_range = speed_range

    def __call__(self, data: Tensor) -> Tensor:
        T = data.shape[0]
        if T < 2:
            return data
        speed = random.uniform(*self.speed_range)
        effective_len = min((T - 1) / speed, T - 1)
        max_start = max(0.0, T - 1 - effective_len)
        start = random.uniform(0, max_start) if max_start > 0 else 0.0
        indices = torch.linspace(start, start + effective_len, T)
        indices = indices.clamp(0, T - 1)
        idx0 = indices.long().clamp(max=T - 2)
        idx1 = idx0 + 1
        frac = (indices - idx0.float()).unsqueeze(-1).unsqueeze(-1)
        return data[idx0] * (1 - frac) + data[idx1] * frac


class PairwiseLandmarkDistances:
    """Compute pairwise Euclidean distances between all landmark pairs.

    This is the core NVIDIA feature representation. Pairwise distances are:
      - Translation invariant (absolute position doesn't matter)
      - Rotation invariant (orientation doesn't matter)
      - Scale-normalized via prior FaceRelativeNormalize (IED = 1)

    Input:  [T, V, 2]  face-relative normalized coordinates
    Output: [T, V*(V-1)//2]  pairwise distances (5565 for V=106)
    """

    def __init__(self, num_vertices: int = 106) -> None:
        self.idx = torch.triu_indices(num_vertices, num_vertices, offset=1)

    def __call__(self, data: Tensor) -> Tensor:
        dist = torch.cdist(data.float(), data.float(), p=2)  # [T, V, V]
        result = dist[:, self.idx[0], self.idx[1]]  # [T, D]
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class SubtractSequenceMean:
    """Remove static face shape by subtracting per-sequence temporal mean."""

    def __call__(self, data: Tensor) -> Tensor:
        return data - data.mean(dim=0, keepdim=True)


class TwoStreamFeatures:
    """Concatenate raw distances with mean-subtracted dynamic distances [and velocity].

    Static stream   preserves the low-motion signal (important for listening).
    Dynamic stream  captures motion deviations (important for cross-generator).
    Velocity stream (feature_mode="three_stream_velocity") is the frame-to-frame
    distance difference; only added when requested, first frame zero-padded.
    Each stream gets its own BatchNorm inside the model, so they're
    independently scaled.
    """

    def __init__(self, feature_mode: str = "two_stream") -> None:
        self.feature_mode = feature_mode

    def __call__(self, data: Tensor) -> Tensor:
        # data: [T, D] pairwise distances
        dyn = data - data.mean(dim=0, keepdim=True)
        streams = [data, dyn]
        if self.feature_mode == "three_stream_velocity":
            vel = torch.zeros_like(data)
            vel[1:] = data[1:] - data[:-1]
            streams.append(vel)
        return torch.cat(streams, dim=-1)  # [T, S*D]


class AddRandomTemporalShuffle:
    """Stack original and temporally-shuffled copy along a new leading axis.

    The shuffled version destroys temporal dynamics while preserving static
    distance patterns. The loss pushes shuffled embeddings away from originals,
    forcing the model to learn temporal motion signatures rather than static
    facial geometry.
    """

    def __call__(self, data: Tensor) -> Tensor:
        T = data.shape[0]
        if T < 2:
            raise ValueError("Need at least 2 frames for temporal shuffling.")
        shuffled = data[torch.randperm(T)]
        return torch.stack((data, shuffled), dim=0)  # [2, T, ...]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrainLandmarkDataset(Dataset):
    """Filtered blosc2 landmark dataset with locally remapped identity labels."""

    def __init__(
        self,
        root: Path,
        meta_path: Path,
        driver_split: Path,
        source_split: Path,
        min_frames: int,
        transform: Callable[[Tensor], Tensor],
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
        id_map: dict[str, int] = {"-1": -1}
        for item in self.meta:
            for raw_key, mapped_key in (
                ("driver_id", "driving_id"),
                ("source_id", "source_index"),
            ):
                raw_id = item[raw_key]
                if raw_id not in id_map:
                    id_map[raw_id] = len(id_map) - 1
                item[mapped_key] = id_map[raw_id]
        self.id_map = id_map
        LOGGER.info(
            "Loaded %d samples and %d identities from %s",
            len(self.meta),
            len(id_map) - 1,
            meta_path,
        )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.meta[index]
        landmarks = blosc2.load_tensor(self.root / item["relpath"])
        if not isinstance(landmarks, Tensor):
            landmarks = torch.as_tensor(landmarks)
        return {
            "landmarks_xyz": self.transform(landmarks),
            "source_id": item["source_index"],
            "driving_id": item["driving_id"],
            "mode": item.get("mode", "speaking"),
            "set": item.get("set", "gallery"),
            "file": (self.root / item["relpath"]).as_posix(),
        }


# ---------------------------------------------------------------------------
# Sampler — includes both self- and cross-reenactments
# ---------------------------------------------------------------------------

class DriverSourceSampler(Sampler[list[int]]):
    """Samples p drivers × ps sources × both modes for contrastive batches.

    Each batch guarantees both speaking and listening clips per driver,
    enabling the pull term to align cross-mode embeddings for free.
    Batch size = p × ps × 2 modes × k_cross.
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
        # Index by (driver, source, mode) for cross-mode sampling
        self.by_dsm: dict[tuple[int, int, str], list[int]] = defaultdict(list)
        self.by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.meta):
            d, s = item["driving_id"], item["source_index"]
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
            raise RuntimeError("Empty sequence for sampling.")
        return self.rng.choice(values, size=count, replace=len(values) < count).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        fallback_count = 0
        for _ in range(self.iterations):
            drivers = self._choice(self.drivers, self.p)
            compatible_sources = [
                s for s in self.sources if all((d, s) in self.by_pair for d in drivers)
            ]
            if len(compatible_sources) >= self.ps:
                sources = self._choice(compatible_sources, self.ps)
            else:
                fallback_count += 1
                sources = self._choice(self.sources, self.ps)

            # Draw from both modes per (driver, source) for cross-mode pull
            flat: list[int] = []
            for d in drivers:
                for s in sources:
                    for mode in self.modes:
                        bucket = self.by_dsm.get((d, s, mode), [])
                        if bucket:
                            flat.extend(self._choice(bucket, self.k_cross))

            usable = len(flat) - (len(flat) % self.world_size)
            yield flat[:usable][self.rank : usable : self.world_size]
        if fallback_count > 0:
            LOGGER.warning(
                "Sampler fallback fired %d/%d iterations (%.1f%%) — "
                "push term may be degraded (sources not shared across drivers)",
                fallback_count, self.iterations,
                100 * fallback_count / self.iterations,
            )

    def __len__(self) -> int:
        return self.iterations


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth: randomly drops the entire residual branch."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class TemporalResBlock(nn.Module):
    """Residual block with 1D temporal convolutions.

    Conv1d(k) → BN → ReLU → Conv1d(k) → BN → + residual → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 1, padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # Zero-init last BN so residual block starts as identity
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(self.drop_path_layer(out) + identity)


class TemporalAttentionPool(nn.Module):
    """Learnable attention pooling over the temporal dimension."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        x_t = x.permute(0, 2, 1)  # [B, T, C]
        weights = self.attn(x_t).squeeze(-1)  # [B, T]
        weights = F.softmax(weights, dim=1)  # [B, T]
        return (x * weights.unsqueeze(1)).sum(dim=2)  # [B, C]


# ---------------------------------------------------------------------------
# Model — Temporal CNN on pairwise landmark distances
# ---------------------------------------------------------------------------

class PairwiseDistanceCNN(nn.Module):
    """NVIDIA-style temporal CNN for dynamic identity embedding.

    Pipeline:
        [B, D=5565, T=71] pairwise distances
        → Conv1d(D, H, 1) input projection
        → Stage 1: temporal ResBlocks (wide kernels, no downsampling)
        → Stage 2: stride-2 downsample + temporal ResBlocks
        → Stage 3: stride-2 downsample + temporal ResBlocks
        → Temporal attention pooling → [B, final_channels]
        → Linear → L2-normalize → [B, embedding_dim]
    """

    def __init__(
        self,
        num_pairs: int,
        embedding_dim: int = 256,
        stage_channels: Sequence[int] = (256, 256, 512),
        stage_blocks: Sequence[int] = (2, 2, 2),
        stage_kernels: Sequence[int] = (7, 5, 3),
        dropout: float = 0.1,
        drop_path: float = 0.1,
        embed_bn: bool = False,
    ) -> None:
        super().__init__()
        # Input batch-norm and projection: D → stage_channels[0]
        self.input_bn = nn.BatchNorm1d(num_pairs)
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_pairs, stage_channels[0], 1, bias=False),
            nn.BatchNorm1d(stage_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Compute linearly increasing drop-path rates
        total_blocks = sum(stage_blocks)
        dp_rates = [drop_path * i / max(1, total_blocks - 1) for i in range(total_blocks)]
        block_idx = 0

        # Build temporal ResNet stages
        stages: list[nn.Module] = []
        in_ch = stage_channels[0]
        for stage_i, (ch, n_blocks, ks) in enumerate(
            zip(stage_channels, stage_blocks, stage_kernels)
        ):
            blocks: list[nn.Module] = []
            for b in range(n_blocks):
                stride = 2 if (b == 0 and stage_i > 0) else 1
                blocks.append(
                    TemporalResBlock(
                        in_ch if b == 0 else ch,
                        ch,
                        kernel_size=ks,
                        stride=stride,
                        drop_path=dp_rates[block_idx],
                    )
                )
                block_idx += 1
                in_ch = ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        # Temporal pooling + embedding projection
        final_ch = stage_channels[-1]
        self.temporal_pool = TemporalAttentionPool(final_ch)
        self.dropout = nn.Dropout(dropout)
        self.embed_proj = nn.Linear(final_ch, embedding_dim)
        self.embed_bn = nn.BatchNorm1d(embedding_dim, affine=False) if embed_bn else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, D, T] — pairwise distances over time
        x = self.input_bn(x)
        x = self.input_proj(x)       # [B, H, T]
        x = self.stages(x)           # [B, final_ch, T']
        x = self.temporal_pool(x)    # [B, final_ch]
        x = self.dropout(x)
        x = self.embed_proj(x)       # [B, embedding_dim]
        x = self.embed_bn(x)
        return F.normalize(x, p=2, dim=-1)


class PairwiseDistanceTransformer(nn.Module):
    """Transformer alternative to PairwiseDistanceCNN — same I/O contract, self-attention
    over the 71-frame sequence instead of a temporal ResNet. Same input features (pairwise
    distances), only the backbone differs; lets architecture and feature changes be tested
    independently. Drop-in replacement: [B, num_pairs, T] -> L2-normalized [B, embedding_dim].
    """

    def __init__(
        self,
        num_pairs: int,
        embedding_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        embed_bn: bool = False,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(num_pairs)
        self.input_proj = nn.Conv1d(num_pairs, d_model, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_pool = TemporalAttentionPool(d_model)
        self.dropout = nn.Dropout(dropout)
        self.embed_proj = nn.Linear(d_model, embedding_dim)
        self.embed_bn = nn.BatchNorm1d(embedding_dim, affine=False) if embed_bn else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, D, T] — pairwise distances over time
        x = self.input_bn(x)
        x = self.input_proj(x)                 # [B, d_model, T]
        x = x.permute(0, 2, 1)                  # [B, T, d_model]
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(x)                     # [B, T, d_model]
        x = x.permute(0, 2, 1)                  # [B, d_model, T]
        x = self.temporal_pool(x)               # [B, d_model]
        x = self.dropout(x)
        x = self.embed_proj(x)                  # [B, embedding_dim]
        x = self.embed_bn(x)
        return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains an exponential moving average of model parameters."""

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
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
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
# Loss — NVIDIA's 3-term contrastive
# ---------------------------------------------------------------------------

def pairwise_squared_distance(x: Tensor, y: Tensor | None = None) -> Tensor:
    y = x if y is None else y
    return (
        x.square().sum(-1, keepdim=True)
        + y.square().sum(-1).unsqueeze(-2)
        - 2 * x @ y.transpose(-2, -1)
    ).clamp_min_(0)


class NVIDIAContrastiveLoss(nn.Module):
    """3-term contrastive loss from the NVIDIA Avatar Fingerprinting paper.

    Given L2-normalized embeddings for a batch of clips:

        N_i = Σ_{j: same driver, j≠i}  s(c_i, c_j)     pull same-driver
        Q_i = Σ_{j: same source, diff driver}  s(c_i, c_j)   push apart
        R_i = Σ_{j: same driver}  s(c_i, shuffled_j)    push shuffled

    where s(a,b) = exp(-||a-b||² / (scale · τ))

        p_i = N_i / (N_i + Q_i + R_i)
        L   = -Σ_i log(p_i)

    Computed in log-space via logsumexp for numerical stability.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: Tensor,
        shuffled: Tensor | None,
        drivers: Tensor,
        sources: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        B = embeddings.shape[0]

        # Pairwise squared distances
        dist = pairwise_squared_distance(embeddings.float())  # [B, B]

        # Normalize for numerical stability
        scale = dist.mean().clamp_min(1e-3)

        # Log-similarities: log s(a,b) = -||a-b||² / (scale · τ)
        log_sim = -dist / (scale * self.temperature)  # [B, B]
        log_sim.fill_diagonal_(-torch.inf)  # exclude self-pairs

        # Identity masks
        same_driver = drivers.unsqueeze(1) == drivers.unsqueeze(0)
        same_source = sources.unsqueeze(1) == sources.unsqueeze(0)
        diag = torch.eye(B, dtype=torch.bool, device=embeddings.device)

        # N: pull — same driver, exclude self
        pull_mask = same_driver & ~diag
        log_N = torch.logsumexp(
            log_sim.masked_fill(~pull_mask, -torch.inf), dim=-1
        )

        # Q: push — same source, different driver
        push_mask = same_source & ~same_driver
        log_Q = torch.logsumexp(
            log_sim.masked_fill(~push_mask, -torch.inf), dim=-1
        )

        # R: time-shuffle push
        logits = [log_N, log_Q]
        if shuffled is not None:
            dist_sh = pairwise_squared_distance(embeddings.float(), shuffled.float())
            log_sim_sh = -dist_sh / (scale * self.temperature)
            # Push away shuffled clips of the same driver
            log_R = torch.logsumexp(
                log_sim_sh.masked_fill(~same_driver, -torch.inf), dim=-1
            )
            logits.append(log_R)

        # p = N / (N + Q + R),  loss = -log(p)
        log_p = F.log_softmax(torch.stack(logits, dim=0), dim=0)[0]  # [B]

        # NaN guard
        finite = torch.isfinite(log_p)
        if finite.all():
            loss = -log_p.mean()
        elif finite.any():
            loss = -log_p[finite].mean()
        else:
            loss = torch.zeros(
                (), device=embeddings.device, dtype=embeddings.dtype, requires_grad=True
            )

        info = {
            "mean_dist": dist.mean().detach(),
            "pull": log_N.exp().mean().detach(),
            "push": log_Q.exp().mean().detach(),
            "p": log_p.exp().mean().detach(),
        }
        return loss, info


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


# ---------------------------------------------------------------------------
# Evaluation — sliding-window
# ---------------------------------------------------------------------------

def _sliding_window_embed(
    model: nn.Module,
    pairwise_dists: Tensor,
    device: torch.device,
    frames: int,
    stride: int,
) -> Tensor:
    """Embed a variable-length pairwise distance sequence using sliding windows."""
    # pairwise_dists: [1, T, D]
    T = pairwise_dists.shape[1]
    windows: list[Tensor] = []
    if T <= frames:
        windows.append(pairwise_dists)
    else:
        for start in range(0, T - frames + 1, stride):
            windows.append(pairwise_dists[:, start : start + frames, :])
        if (T - frames) % stride != 0:
            windows.append(pairwise_dists[:, T - frames : T, :])

    embs: list[Tensor] = []
    for w in windows:
        x = w.to(device, non_blocking=True).permute(0, 2, 1).contiguous().float()
        emb = model(x)
        embs.append(emb)

    avg_emb = torch.stack(embs).mean(dim=0)
    return F.normalize(avg_emb, p=2, dim=-1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    distance_metric: str = "cosine",
    frames: int = 71,
    window_stride: int = 35,
    embedding_dim: int = 256,
) -> dict[str, float]:
    model.eval()
    buckets: dict[tuple[str, str], dict[str, list[Tensor]]] = {
        ("probe", "speaking"): {"emb": [], "driving": []},
        ("probe", "listening"): {"emb": [], "driving": []},
        ("gallery", "speaking"): {"emb": [], "driving": []},
        ("gallery", "listening"): {"emb": [], "driving": []},
    }

    for batch in loader:
        raw = batch["landmarks_xyz"]  # [1, T, D]
        embedding = _sliding_window_embed(model, raw, device, frames, window_stride)
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
        b["emb"] = torch.stack(b["emb"]) if b["emb"] else torch.empty(0, embedding_dim)
        b["driving"] = torch.stack(b["driving"]).long() if b["driving"] else torch.empty(0).long()

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


def compute_official_score(metrics: dict[str, float]) -> float:
    """Mean of 3 conditions: speaking-speaking, listening-listening, cross."""
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

def build_loaders(
    config: Settings, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader]:
    normalize = FaceRelativeNormalize()
    pairwise = PairwiseLandmarkDistances(config.vertices)

    # Train transform: keep data as landmarks [T, V, 2] — pairwise distances
    # are computed on GPU to offload CPU and allow data workers.
    train_dataset = TrainLandmarkDataset(
        config.data_root,
        config.meta_path,
        config.train_driver_split,
        config.train_source_split,
        config.frames,
        Compose([
            PadShortClips(config.frames),
            RandomClipWithDrop(config.frames, 0.05),
            normalize,
            SpatialAugment(),
            TemporalSpeedPerturb(),
            AddRandomTemporalShuffle(),
        ]),
    )
    # Val transform: batch_size=1, CPU pairwise distances are fine.
    val_dataset = TrainLandmarkDataset(
        config.data_root,
        config.meta_path,
        config.val_driver_split,
        config.val_source_split,
        config.frames,
        Compose([
            PadShortClips(config.frames),
            normalize,
            pairwise,
            TwoStreamFeatures(config.feature_mode),
        ]),
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
# Training
# ---------------------------------------------------------------------------

def _gpu_pairwise_distances(
    landmarks: Tensor,
    triu_idx: Tensor,
    chunk_size: int = 32,
    feature_mode: str = "two_stream",
) -> Tensor:
    """Pairwise distances on GPU: static ∥ dynamic [∥ velocity].

    Static stream   = raw IED-normalised pairwise distances (face shape + motion).
    Dynamic stream  = distances minus per-sequence temporal mean (motion only).
    Velocity stream = frame-to-frame distance difference (first frame zero-padded).
    Only added when feature_mode == "three_stream_velocity"; discards less absolute
    geometry than the other two streams, on the hypothesis that it's more
    renderer-invariant (see docs/video_ablation_study.md diff-generator discussion).

    Concatenated along feature dim → [B, S*D, T].  Chunked to cap peak memory.
    """
    B = landmarks.shape[0]
    chunks: list[Tensor] = []
    for start in range(0, B, chunk_size):
        lm = landmarks[start : start + chunk_size]
        pa = lm[:, :, triu_idx[0], :]          # [chunk, T, D, 2]
        pb = lm[:, :, triu_idx[1], :]
        d = (pa - pb).norm(dim=-1)              # [chunk, T, D]
        del pa, pb
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        dyn = d - d.mean(dim=1, keepdim=True)   # remove static shape
        streams = [d, dyn]
        if feature_mode == "three_stream_velocity":
            vel = torch.zeros_like(d)
            vel[:, 1:] = d[:, 1:] - d[:, :-1]
            streams.append(vel)
        feats = torch.cat(streams, dim=-1)      # [chunk, T, S*D]
        chunks.append(feats.permute(0, 2, 1).contiguous())  # [chunk, S*D, T]
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]


def _build_model(config: Settings) -> nn.Module:
    num_pairs = config.feature_streams * config.num_pairs
    if config.arch == "transformer":
        return PairwiseDistanceTransformer(
            num_pairs=num_pairs,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            embed_bn=config.embed_bn,
        )
    return PairwiseDistanceCNN(
        num_pairs=num_pairs,
        embedding_dim=config.embedding_dim,
        stage_channels=config.stage_channels,
        stage_blocks=config.stage_blocks,
        stage_kernels=config.stage_kernels,
        dropout=config.dropout,
        drop_path=config.drop_path,
        embed_bn=config.embed_bn,
    )


def train(config: Settings, resume: Path | None = None) -> None:
    rank, world_size, local_rank = distributed_context()
    setup_logging(config.output_dir)
    seed_everything(config.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config, rank, world_size)

    # Tracking setup
    csv_logger: CSVMetricsLogger | None = None
    if is_main_process():
        setup_mlflow_run("nvidia-tcnn", "nvidia-tcnn-train", config)
        csv_logger = CSVMetricsLogger(config.output_dir)

    raw_model = _build_model(config).to(device)

    model: nn.Module = raw_model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    criterion = NVIDIAContrastiveLoss(config.temperature).to(device)
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, config.iterations, config.min_lr
    )
    amp_enabled = config.amp and device.type == "cuda"

    ema: ModelEMA | None = None
    if config.ema_decay > 0:
        ema = ModelEMA(raw_model, config.ema_decay)

    start = 0
    if resume:
        start = load_checkpoint(resume, model, optimizer, scheduler, ema)

    best_score = -math.inf
    patience_counter = 0
    loader_iterator = iter(train_loader)
    triu_idx = torch.triu_indices(
        config.vertices, config.vertices, offset=1, device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Model parameters: %d total, %d trainable", total_params, trainable)
    LOGGER.info("Input features: %d pairwise distances per frame", config.num_pairs)

    for iteration in range(start + 1, config.iterations + 1):
        model.train()
        batch = next(loader_iterator)

        # [B, 2, T, V, C] → [2*B, T, V, C] → GPU pairwise dists → [2*B, D, T]
        landmarks = (
            batch["landmarks_xyz"]
            .to(device, non_blocking=True)
            .float()
            .reshape(-1, config.frames, config.vertices, 2)
        )
        features = _gpu_pairwise_distances(landmarks, triu_idx, feature_mode=config.feature_mode)
        del landmarks
        driving = batch["driving_id"].to(device, non_blocking=True).long()
        source = batch["source_id"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16,
        ):
            embeddings = model(features)  # [2*B, embedding_dim]

        # Split original / shuffled
        emb_float = embeddings.float()
        original, shuffled = emb_float.reshape(-1, 2, config.embedding_dim).unbind(1)

        loss, info = criterion(original, shuffled, driving, source)

        if not torch.isfinite(loss):
            LOGGER.warning("Non-finite loss at iteration %d — skipping", iteration)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        loss.backward()
        if config.grad_clip > 0:
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
                "iteration %d/%d | loss %.6f | p %.4f | lr %.2e%s",
                iteration,
                config.iterations,
                loss.item(),
                info["p"].item(),
                optimizer.param_groups[0]["lr"],
                grad_str,
            )
            train_metrics = {
                "loss": loss.item(),
                "p": info["p"].item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            mlflow.log_metrics(
                {f"train/{k}": v for k, v in train_metrics.items()},
                step=iteration,
            )
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
                    embedding_dim=config.embedding_dim,
                )
                # Compute scores before logging so CSV + mlflow both get them
                live_score = compute_official_score(metrics)
                metrics["score"] = live_score

                if ema is not None:
                    ema_metrics = evaluate(
                        ema.module, val_loader, device,
                        distance_metric=config.eval_distance,
                        frames=config.frames,
                        window_stride=config.window_stride,
                        embedding_dim=config.embedding_dim,
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
                    "validation at %d | score %.6f (%s) | %s",
                    iteration, current_score, score_source,
                    ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()),
                )
                mlflow.log_metrics(
                    {f"val/{k}": v for k, v in metrics.items()},
                    step=iteration,
                )
                csv_logger.log("val", metrics, step=iteration)
                save_checkpoint(
                    config.output_dir / "last_model.pt",
                    model, optimizer, scheduler, iteration, metrics, ema,
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
                            },
                            config.output_dir / "best_model.pt",
                        )
                    else:
                        save_checkpoint(
                            config.output_dir / "best_model.pt",
                            model, optimizer, scheduler, iteration, metrics, ema,
                        )
                    LOGGER.info(
                        "New best model (%s) at iteration %d with score %.6f",
                        score_source, iteration, best_score,
                    )
                    mlflow.log_metric("val/best_score", best_score, step=iteration)
                else:
                    patience_counter += 1
                    if config.patience > 0 and patience_counter >= config.patience:
                        LOGGER.info(
                            "Early stopping at iteration %d "
                            "(no improvement for %d validations)",
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
    model = _build_model(config).to(device)
    load_checkpoint(checkpoint, model)
    metrics = evaluate(
        model, val_loader, device,
        distance_metric=config.eval_distance,
        frames=config.frames,
        window_stride=config.window_stride,
        embedding_dim=config.embedding_dim,
    )
    score = compute_official_score(metrics)
    LOGGER.info(
        "test | score=%.6f | %s",
        score,
        ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
    )
    if distributed.is_initialized():
        distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test", action="store_true", help="Evaluate a checkpoint instead of training."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CONFIG.output_dir / "best_model.pt",
        help="Checkpoint for --test or to resume training.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from --checkpoint."
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-drivers", type=int, default=None)
    parser.add_argument("--batch-sources", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--drop-path", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None, help="0 to disable")
    parser.add_argument("--patience", type=int, default=None, help="0 to disable")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--embed-bn", action="store_true",
                        help="BatchNorm on the embedding (training statistics, frozen at eval)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--eval-distance", type=str, choices=["cosine", "euclidean"], default=None
    )
    parser.add_argument(
        "--feature-mode", type=str,
        choices=["two_stream", "three_stream_velocity"], default=None,
    )
    parser.add_argument("--arch", type=str, choices=["cnn", "transformer"], default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    _cli_map = {
        "lr": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_drivers": "batch_drivers",
        "batch_sources": "batch_sources",
        "drop_path": "drop_path",
        "ema_decay": "ema_decay",
        "warmup_steps": "warmup_steps",
        "window_stride": "window_stride",
        "eval_distance": "eval_distance",
        "feature_mode": "feature_mode",
        "arch": "arch",
    }
    for cli_attr, field_name in _cli_map.items():
        val = getattr(args, cli_attr, None)
        if val is not None:
            overrides[field_name] = val
    if getattr(args, "embed_bn", False):
        overrides["embed_bn"] = True
    for name in ("iterations", "seed", "dropout", "patience", "temperature"):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val

    config = replace(CONFIG, **overrides) if overrides else CONFIG

    if args.test:
        test(config, args.checkpoint)
    else:
        train(config, args.checkpoint if args.resume else None)


if __name__ == "__main__":
    main()
