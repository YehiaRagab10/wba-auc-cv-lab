#!/usr/bin/env python3
"""Self-contained training and testing program for WBA2026 2D landmark fingerprints.

Upgraded from baseline: all 106 landmarks, anatomical graph with 4 subsets,
6-channel input (position + velocity + acceleration), region node embeddings,
temporal attention pooling, L2-normalized embeddings, face-relative normalization,
spatial augmentation, gradient clipping, EER metric, EMA, cosine LR schedule,
early stopping, sliding-window evaluation, stochastic depth.

Examples:
        python trainer.py
        python trainer.py --test --checkpoint output/.../best_model.pt
        python trainer.py --lr 1e-4 --weight-decay 0.05 --seed 42
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
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

import sys
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from tracking import CSVMetricsLogger, setup_mlflow_run


# ---------------------------------------------------------------------------
# Semantic landmark regions (InsightFace 2D-106 convention)
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

# Which regions are anatomically adjacent (for inter-region graph subset)
REGION_ADJACENCY = [
    ("face_contour", "left_eyebrow"),
    ("face_contour", "right_eyebrow"),
    ("face_contour", "mouth"),
    ("left_eye", "left_eyebrow"),
    ("right_eye", "right_eyebrow"),
    ("left_eye", "nose"),
    ("right_eye", "nose"),
    ("nose", "mouth"),
    ("left_eye", "right_eye"),
    ("left_eyebrow", "right_eyebrow"),
]


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
    output_dir: Path = PROJECT_ROOT / "output/landmark_2d_106/stgcn"
    frames: int = 75
    vertices: int = 106
    channels: int = 6           # x, y, dx/dt, dy/dt, d²x/dt², d²y/dt²
    region_embed_dim: int = 16
    embedding_dim: int = 256
    blocks: tuple[int, ...] = (2, 2, 2)
    batch_drivers: int = 4 
    batch_sources: int = 2
    cross_samples: int = 2
    iterations: int = 30_000
    validation_interval: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    drop_path: float = 0.1
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    patience: int = 10          # early stopping patience (in validation intervals, 0=off)
    ema_decay: float = 0.999    # EMA decay rate (0=off)
    window_stride: int = 37     # sliding-window stride for eval/inference
    graph_loss_weight: float = 0.01
    embed_bn: bool = False      # BatchNorm on the embedding (training statistics, frozen at eval)
    temperature: float = 0.5
    workers: int = 0
    seed: int = 0
    amp: bool = True
    eval_distance: str = "cosine"


CONFIG = Settings()


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


LOGGER = logging.getLogger("trainer")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Reseed each DataLoader worker to avoid duplicate random states."""
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
# Graph construction — anatomical multi-subset adjacency
# ---------------------------------------------------------------------------

def make_anatomical_graph(num_vertices: int = 106) -> Tensor:
    """Build a 4-subset adjacency matrix from semantic landmark regions.

    Subsets:
        0 — identity (self-loops)
        1 — local topology (sequential within each region, closed for eyes/mouth)
        2 — intra-region (fully connected within each region)
        3 — inter-region (connections between anatomically adjacent regions)
    """
    identity = torch.eye(num_vertices)

    # Subset 1: local sequential connections
    local = torch.zeros(num_vertices, num_vertices)
    closed_regions = {"left_eye", "right_eye", "mouth"}
    for name, indices in LANDMARK_REGIONS.items():
        for k in range(len(indices) - 1):
            i, j = indices[k], indices[k + 1]
            local[i, j] = 1
            local[j, i] = 1
        if name in closed_regions:
            local[indices[0], indices[-1]] = 1
            local[indices[-1], indices[0]] = 1

    # Subset 2: intra-region fully connected
    intra = torch.zeros(num_vertices, num_vertices)
    for indices in LANDMARK_REGIONS.values():
        for i in indices:
            for j in indices:
                if i != j:
                    intra[i, j] = 1

    # Subset 3: inter-region connections
    inter = torch.zeros(num_vertices, num_vertices)
    for r1_name, r2_name in REGION_ADJACENCY:
        for i in LANDMARK_REGIONS[r1_name]:
            for j in LANDMARK_REGIONS[r2_name]:
                inter[i, j] = 1
                inter[j, i] = 1

    def normalize(g: Tensor) -> Tensor:
        return g / g.sum(dim=0, keepdim=True).clamp_min(1)

    return torch.stack([identity, normalize(local), normalize(intra), normalize(inter)])


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
    """Pad short sequences by repeating the last frame."""

    def __init__(self, min_clip_length: int) -> None:
        self.min_clip_length = min_clip_length

    def __call__(self, data: Tensor) -> Tensor:
        if data.shape[0] < self.min_clip_length:
            pad_size = self.min_clip_length - data.shape[0]
            last_frame = data[-1:].expand(pad_size, *data.shape[1:])
            return torch.cat([data, last_frame], dim=0)
        return data


class RandomClipWithDrop:
    def __init__(self, clip_length: int, drop_rate: float) -> None:
        self.clip_length = clip_length
        self.drop_rate = drop_rate

    def __call__(self, data: Tensor) -> Tensor:
        if data.shape[0] < self.clip_length:
            raise ValueError(
                f"Expected at least {self.clip_length} frames, got {data.shape[0]}."
            )
        if data.shape[0] == self.clip_length:
            return data
        dynamic_length = min(
            int(self.clip_length * (1 + self.drop_rate)), data.shape[0]
        )
        start = torch.randint(data.shape[0] - dynamic_length + 1, ()).item()
        indices = (
            torch.randperm(dynamic_length)[: self.clip_length].sort().values + start
        )
        return data[indices]


class ClipCenter:
    def __init__(self, clip_length: int) -> None:
        self.clip_length = clip_length

    def __call__(self, data: Tensor) -> Tensor:
        if data.shape[0] < self.clip_length:
            raise ValueError(
                f"Expected at least {self.clip_length} frames, got {data.shape[0]}."
            )
        start = (data.shape[0] - self.clip_length) // 2
        return data[start : start + self.clip_length]


class FaceRelativeNormalize:
    """Center on face centroid, align eyes horizontally, scale by inter-eye distance."""

    LEFT_EYE = list(range(33, 43))
    RIGHT_EYE = list(range(87, 97))

    def __call__(self, data: Tensor) -> Tensor:
        data = data.float()
        # Center on per-frame centroid
        centroid = data.mean(dim=-2, keepdim=True)
        centered = data - centroid
        # Correct in-plane rotation (roll) so eyes are horizontal
        left_center = centered[:, self.LEFT_EYE, :].mean(dim=1)    # [T, 2]
        right_center = centered[:, self.RIGHT_EYE, :].mean(dim=1)  # [T, 2]
        diff = right_center - left_center                           # [T, 2]
        angle = torch.atan2(diff[:, 1], diff[:, 0])                 # [T]
        cos_a = torch.cos(-angle).unsqueeze(1).unsqueeze(2)         # [T, 1, 1]
        sin_a = torch.sin(-angle).unsqueeze(1).unsqueeze(2)         # [T, 1, 1]
        x = centered[..., 0:1]
        y = centered[..., 1:2]
        rotated = torch.cat(
            [x * cos_a - y * sin_a, x * sin_a + y * cos_a], dim=-1
        )
        # Scale by inter-eye distance
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
    """Randomly resample frames at perturbed speed (output keeps same frame count)."""

    def __init__(self, speed_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.speed_range = speed_range

    def __call__(self, data: Tensor) -> Tensor:
        # data: [T, V, C]
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


class AddTemporalDerivatives:
    """Append velocity and acceleration channels: [T, V, 2] -> [T, V, 6]."""

    def __call__(self, data: Tensor) -> Tensor:
        velocity = torch.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        velocity[0] = velocity[1]

        acceleration = torch.zeros_like(data)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = acceleration[1]

        velocity = velocity.clamp(-10, 10)
        acceleration = acceleration.clamp(-10, 10)
        return torch.cat([data, velocity, acceleration], dim=-1)


class AddRandomTemporalShuffle:
    """Stack each clip and a temporally shuffled copy along a new leading axis."""

    def __call__(self, data: Tensor) -> Tensor:
        if data.shape[-3] < 2:
            raise ValueError("At least two frames are required for temporal shuffling.")
        shuffled = data.index_select(
            -3, torch.randperm(data.shape[-3], device=data.device)
        )
        return torch.stack((data, shuffled), dim=-4)


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
# Sampler
# ---------------------------------------------------------------------------

class SourceDriverCrossSampler(Sampler[list[int]]):
    """Samples p drivers crossed with ps sources, as required by LOGSOFT loss."""

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
        self.by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.meta):
            if item["source_index"] != item["driving_id"]:
                self.by_pair[item["driving_id"], item["source_index"]].append(index)
        self.drivers = sorted({driver for driver, _ in self.by_pair})
        self.sources = sorted({source for _, source in self.by_pair})
        if not self.drivers or not self.sources:
            raise RuntimeError("The training split has no cross-reenactment samples.")

    def _choice(self, values: Sequence[int], count: int) -> list[int]:
        if not values:
            raise RuntimeError("Requested a source/driver pair which has no samples.")
        return self.rng.choice(values, size=count, replace=len(values) < count).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.iterations):
            drivers = self._choice(self.drivers, self.p)
            compatible_sources = [
                s for s in self.sources if all((d, s) in self.by_pair for d in drivers)
            ]
            if not compatible_sources:
                indices = [
                    self._choice(self.by_pair[d, s], self.k_cross)
                    for d in drivers
                    for s in self.sources
                    if (d, s) in self.by_pair
                ]
                flat = [index for group in indices for index in group]
            else:
                sources = self._choice(compatible_sources, self.ps)
                flat = [
                    index
                    for d in drivers
                    for s in sources
                    for index in self._choice(self.by_pair[d, s], self.k_cross)
                ]
            usable = len(flat) - (len(flat) % self.world_size)
            yield flat[:usable][self.rank : usable : self.world_size]

    def __len__(self) -> int:
        return self.iterations


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

def conv_init(layer: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class DropPath(nn.Module):
    """Stochastic depth: randomly drops the entire residual branch during training."""

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


class UnitTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, 1),
            (stride, 1),
            ((kernel_size - 1) // 2, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()
        conv_init(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))


class UnitGCN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, graph_shape: Sequence[int]
    ) -> None:
        super().__init__()
        self.subsets = graph_shape[0]
        self.inter_channels = max(1, out_channels // 4)
        self.pa = nn.Parameter(torch.full(tuple(graph_shape), 1e-6))
        self.query_key = nn.Conv2d(
            in_channels, 2 * self.subsets * self.inter_channels, 1
        )
        self.projection = nn.Conv2d(
            in_channels * self.subsets, out_channels, 1, bias=False
        )
        self.down = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        for layer in (self.query_key, self.projection):
            conv_init(layer)
        nn.init.constant_(self.bn.weight, 1e-6)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: Tensor, base_graph: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, frames, vertices = x.shape
        qk = self.query_key(x).view(
            batch, 2, self.subsets, self.inter_channels, frames, vertices
        )
        query, key = qk.unbind(1)
        score = torch.einsum("nsctq,nsctk->nsqk", query, key) / (
            self.inter_channels * frames
        )
        attention = score.softmax(dim=-2) + self.pa.clamp(-1, 1) + base_graph
        propagated = torch.einsum("nsqv,nctq->nsctv", attention, x).reshape(
            batch, self.subsets * channels, frames, vertices
        )
        return (
            self.activation(self.bn(self.projection(propagated)) + self.down(x)),
            attention,
        )


class TCN_GCN_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        graph_shape: Sequence[int],
        stride: int = 1,
        residual: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, graph_shape)
        self.tcn = UnitTCN(out_channels, out_channels, stride=stride)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path_layer = (
            DropPath(drop_path) if drop_path > 0 and residual else nn.Identity()
        )
        self.residual = (
            nn.Identity()
            if residual and in_channels == out_channels and stride == 1
            else (
                UnitTCN(in_channels, out_channels, 1, stride, activation=False)
                if residual
                else None
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, graph: Tensor) -> tuple[Tensor, Tensor]:
        y, attention = self.gcn(x, graph)
        y = self.drop(self.tcn(y))
        if self.residual is not None:
            return (
                self.activation(self.drop_path_layer(y) + self.residual(x)),
                attention,
            )
        return self.activation(y), attention


class RegionNodeEmbedding(nn.Module):
    """Learnable per-node embedding based on which semantic region a node belongs to."""

    def __init__(
        self,
        num_regions: int,
        embed_dim: int,
        num_vertices: int = 106,
        regions: dict[str, list[int]] | None = None,
    ) -> None:
        super().__init__()
        if regions is None:
            regions = LANDMARK_REGIONS
        self.embedding = nn.Embedding(num_regions, embed_dim)
        region_ids = torch.zeros(num_vertices, dtype=torch.long)
        for idx, (_, indices) in enumerate(regions.items()):
            for i in indices:
                region_ids[i] = idx
        self.register_buffer("region_ids", region_ids)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, V]
        emb = self.embedding(self.region_ids)         # [V, embed_dim]
        emb = emb.T.unsqueeze(0).unsqueeze(2)         # [1, embed_dim, 1, V]
        emb = emb.expand(x.shape[0], -1, x.shape[2], -1)
        return torch.cat([x, emb], dim=1)             # [B, C+embed_dim, T, V]


class TemporalAttentionPool(nn.Module):
    """Learnable attention pooling over time (after mean-pooling vertices)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, V]
        x_v = x.mean(dim=3)                           # [B, C, T]
        x_t = x_v.permute(0, 2, 1)                    # [B, T, C]
        weights = self.attn(x_t).squeeze(-1)           # [B, T]
        weights = F.softmax(weights, dim=1)            # [B, T]
        return (x_v * weights.unsqueeze(1)).sum(dim=2) # [B, C]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LandmarkSTGCN(nn.Module):
    """ST-GCN with region embedding, temporal attention pooling, L2-normalised output."""

    def __init__(
        self,
        graph: Tensor,
        channels: int,
        dim: int,
        out_dim: int,
        blocks: Sequence[int],
        vertices: int,
        region_embed_dim: int = 16,
        embed_bn: bool = False,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_bn = nn.BatchNorm1d(channels * vertices)
        self.embed_bn = nn.BatchNorm1d(out_dim, affine=False) if embed_bn else nn.Identity()
        self.region_embed = RegionNodeEmbedding(
            len(LANDMARK_REGIONS), region_embed_dim, vertices
        )
        input_channels = channels + region_embed_dim
        base_dim = dim // 2 ** (len(blocks) - 1)
        graph_shape = tuple(graph.shape)

        # Linearly increasing drop_path rate across blocks (stochastic depth)
        total_blocks = sum(blocks) + len(blocks)
        block_idx = 0

        def _dp_rate() -> float:
            nonlocal block_idx
            rate = drop_path * block_idx / max(1, total_blocks - 1)
            block_idx += 1
            return rate

        layers: list[nn.Module] = [
            TCN_GCN_Block(
                input_channels, base_dim, graph_shape,
                residual=False, dropout=dropout, drop_path=_dp_rate(),
            )
        ]
        for level, count in enumerate(blocks):
            if level:
                layers.append(
                    TCN_GCN_Block(
                        base_dim * 2 ** (level - 1),
                        base_dim * 2**level,
                        graph_shape,
                        stride=2,
                        dropout=dropout,
                        drop_path=_dp_rate(),
                    )
                )
            layers.extend(
                TCN_GCN_Block(
                    base_dim * 2**level, base_dim * 2**level, graph_shape,
                    dropout=dropout, drop_path=_dp_rate(),
                )
                for _ in range(count)
            )
        final_dim = base_dim * 2 ** (len(blocks) - 1)
        if final_dim != out_dim:
            layers.append(
                TCN_GCN_Block(
                    final_dim, out_dim, graph_shape,
                    dropout=dropout, drop_path=_dp_rate(),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.temporal_pool = TemporalAttentionPool(out_dim)
        self.register_buffer("graph", graph)

    def forward(self, landmarks: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, frames, vertices = landmarks.shape
        # Batch-norm on raw input
        x = landmarks.permute(0, 1, 3, 2).reshape(batch, channels * vertices, frames)
        x = (
            self.data_bn(x)
            .reshape(batch, channels, vertices, frames)
            .permute(0, 1, 3, 2)
        )
        # Inject region identity
        x = self.region_embed(x)
        # ST-GCN blocks
        attention: Tensor | None = None
        for layer in self.layers:
            x, attention = layer(x, self.graph)
        assert attention is not None
        # Temporal attention pooling + L2 normalise
        embedding = self.temporal_pool(x)
        embedding = self.embed_bn(embedding)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding, attention


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average of model weights)
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
# Loss
# ---------------------------------------------------------------------------

def pairwise_squared_distance(x: Tensor, y: Tensor | None = None) -> Tensor:
    y = x if y is None else y
    return (
        x.square().sum(-1, keepdim=True)
        + y.square().sum(-1).unsqueeze(-2)
        - 2 * x @ y.transpose(-2, -1)
    ).clamp_min_(0)


class DynamicIdentityLoss(nn.Module):
    """Numerically stable LOGSOFT identity loss, including temporal-shuffle negatives."""

    def __init__(
        self, temperature: float, norm_dist: str = "mean", match_selection: str = "max"
    ) -> None:
        super().__init__()
        self.temperature, self.norm_dist, self.match_selection = (
            temperature,
            norm_dist,
            match_selection,
        )

    def _reduce_time(
        self, values: Tensor, batch: int, frames: int, other_batch: int
    ) -> Tensor:
        values = values.reshape(batch, frames, other_batch, -1)
        if self.match_selection == "max":
            return values.amax(dim=-1)
        if self.match_selection == "min":
            return values.amin(dim=-1)
        return values.mean(dim=-1)

    def forward(
        self,
        embedding: Tensor,
        shuffled: Tensor | None,
        driving: Tensor,
        source: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, dimensions, frames = embedding.shape
        flat = embedding.permute(0, 2, 1).reshape(batch * frames, dimensions).float()
        distance = pairwise_squared_distance(flat)
        scale = (
            distance.mean().clamp_min(1e-3)
            if self.norm_dist == "mean"
            else distance.amax().clamp_min(1e-3)
        )
        similarity = distance / (-scale * self.temperature)
        if self.match_selection == "max":
            similarity.fill_diagonal_(-torch.inf)
        elif self.match_selection == "min":
            similarity.fill_diagonal_(torch.inf)
        similarity = self._reduce_time(similarity, batch, frames, batch)
        same_driver = driving[:, None].eq(driving[None, :])
        same_source = source[:, None].eq(source[None, :])
        log_pull = torch.logsumexp(
            similarity.masked_fill(~same_driver[:, None], -torch.inf), dim=-1
        )
        log_push = torch.logsumexp(
            similarity.masked_fill(~(same_source & ~same_driver)[:, None], -torch.inf),
            dim=-1,
        )
        logits = [log_pull, log_push]
        distance_shuffled = None
        if shuffled is not None:
            shuffled_flat = (
                shuffled.permute(0, 2, 1).reshape(batch * frames, dimensions).float()
            )
            distance_shuffled = pairwise_squared_distance(flat, shuffled_flat)
            shuffled_similarity = self._reduce_time(
                distance_shuffled / (-scale * self.temperature), batch, frames, batch
            )
            log_shuffle = torch.logsumexp(
                shuffled_similarity.masked_fill(~same_driver[:, None], -torch.inf),
                dim=-1,
            )
            logits.append(log_shuffle)
        log_probability = F.log_softmax(torch.stack(logits), dim=0)[0]
        # Guard against NaN/inf from degenerate batches (e.g. driver with no pairs)
        finite = torch.isfinite(log_probability)
        if finite.all():
            loss = -log_probability.mean()
        elif finite.any():
            loss = -log_probability[finite].mean()
        else:
            loss = torch.zeros((), device=log_probability.device, dtype=log_probability.dtype, requires_grad=True)
        info = {
            "mean_dist": distance.mean().detach(),
            "pull": log_pull.exp().mean().detach(),
            "push": log_push.exp().mean().detach(),
            "p": log_probability.exp().mean().detach(),
        }
        if distance_shuffled is not None:
            info["mean_dist_sh"] = distance_shuffled.mean().detach()
        return loss, info


class FingerprintingLoss(nn.Module):
    def __init__(self, temperature: float, graph_weight: float) -> None:
        super().__init__()
        self.identity_loss = DynamicIdentityLoss(temperature)
        self.graph_weight = graph_weight

    def forward(
        self, embeddings: Tensor, graphs: Tensor, driving: Tensor, source: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if embeddings.shape[0] % 2:
            raise ValueError(
                "Training embeddings must contain original/shuffled pairs."
            )
        original_embedding, shuffled_embedding = embeddings.reshape(
            -1, 2, embeddings.shape[-1]
        ).unbind(1)
        original_graph, shuffled_graph = (
            graphs.mean(dim=1)
            .reshape(-1, 2, graphs.shape[-2] * graphs.shape[-1])
            .unbind(1)
        )
        embedding_loss, info = self.identity_loss(
            original_embedding.unsqueeze(-1),
            shuffled_embedding.unsqueeze(-1),
            driving,
            source,
        )
        graph_loss, graph_info = self.identity_loss(
            original_graph.unsqueeze(-1), shuffled_graph.unsqueeze(-1), driving, source
        )
        info.update({f"graph_{key}": value for key, value in graph_info.items()})
        info["fingerprinting_loss"] = embedding_loss.detach()
        info["graph_loss"] = graph_loss.detach()
        return embedding_loss + self.graph_weight * graph_loss, info


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
    """Compute Equal Error Rate from similarity scores and binary targets."""
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
    eer = fpr[i].item() * (1 - alpha) + fpr[i + 1].item() * alpha
    return eer


def _compute_pair_metrics(
    probe_emb: Tensor,
    probe_ids: Tensor,
    gallery_emb: Tensor,
    gallery_ids: Tensor,
    distance_metric: str = "cosine",
) -> tuple[float, float]:
    """Compute AUC and EER between probe and gallery sets."""
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
# Evaluation — sliding-window with official 3-condition scoring
# ---------------------------------------------------------------------------

def _sliding_window_embed(
    model: nn.Module,
    raw_landmarks: Tensor,
    device: torch.device,
    frames: int,
    stride: int,
) -> tuple[Tensor, Tensor]:
    """Embed a variable-length sequence using sliding windows, then average."""
    # raw_landmarks: [1, T, V, C]
    T = raw_landmarks.shape[1]
    windows: list[Tensor] = []
    if T <= frames:
        windows.append(raw_landmarks)
    else:
        for start in range(0, T - frames + 1, stride):
            windows.append(raw_landmarks[:, start : start + frames, :, :])
        # Cover the tail if stride doesn't divide evenly
        if (T - frames) % stride != 0:
            windows.append(raw_landmarks[:, T - frames : T, :, :])

    embs: list[Tensor] = []
    graphs: list[Tensor] = []
    for w in windows:
        x = w.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float()
        emb, graph = model(x)
        embs.append(emb)
        graphs.append(graph.mean(dim=1).flatten(1))

    avg_emb = torch.stack(embs).mean(dim=0)
    avg_emb = F.normalize(avg_emb, p=2, dim=-1)
    avg_graph = torch.stack(graphs).mean(dim=0)
    return avg_emb, avg_graph


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    distance_metric: str = "cosine",
    frames: int = 75,
    window_stride: int = 37,
) -> dict[str, float]:
    model.eval()
    buckets: dict[tuple[str, str], dict[str, list[Tensor]]] = {
        ("probe", "speaking"): {"emb": [], "graph": [], "driving": []},
        ("probe", "listening"): {"emb": [], "graph": [], "driving": []},
        ("gallery", "speaking"): {"emb": [], "graph": [], "driving": []},
        ("gallery", "listening"): {"emb": [], "graph": [], "driving": []},
    }

    for batch in loader:
        raw = batch["landmarks_xyz"]  # [1, T, V, C] (batch_size=1)
        embedding, graph = _sliding_window_embed(
            model, raw, device, frames, window_stride
        )

        modes = batch["mode"]
        sets = batch["set"]
        driving = batch["driving_id"].cpu()

        for i in range(len(modes)):
            key = (sets[i], modes[i])
            if key in buckets:
                buckets[key]["emb"].append(embedding[i].cpu())
                buckets[key]["graph"].append(graph[i].cpu())
                buckets[key]["driving"].append(driving[i])
    for key in buckets:
        b = buckets[key]
        b["emb"] = torch.stack(b["emb"]) if b["emb"] else torch.empty(0, 256)
        b["graph"] = (
            torch.stack(b["graph"])
            if b["graph"]
            else torch.empty(0, 106 * 106)
        )
        b["driving"] = torch.stack(b["driving"]).long()

    output: dict[str, float] = {}
    for dist_name, attr in [("emb", "emb"), ("graph", "graph")]:
        for probe_mode in ("speaking", "listening"):
            for gallery_mode in ("speaking", "listening"):
                p = buckets[("probe", probe_mode)]
                g = buckets[("gallery", gallery_mode)]
                if p["emb"].numel() == 0 or g["emb"].numel() == 0:
                    continue
                auc, eer = _compute_pair_metrics(
                    p[attr], p["driving"], g[attr], g["driving"], distance_metric
                )
                tag = f"{dist_name}_{probe_mode}_vs_{gallery_mode}"
                output[f"{tag}_auc"] = auc
                output[f"{tag}_eer"] = eer

    return output


def compute_official_score(metrics: dict[str, float]) -> float:
    """Compute official WBA2026 score: mean of 3 conditions.

    Conditions:
        1. Speaking vs Speaking
        2. Non-speaking (listening) vs Non-speaking
        3. Cross-condition: mean(speaking_vs_listening, listening_vs_speaking)
    """
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
    common_pre = [
        PadShortClips(config.frames),
    ]
    normalize = FaceRelativeNormalize()
    derivatives = AddTemporalDerivatives()

    train_dataset = TrainLandmarkDataset(
        config.data_root,
        config.meta_path,
        config.train_driver_split,
        config.train_source_split,
        config.frames,
        Compose(
            [
                *common_pre,
                RandomClipWithDrop(config.frames, 0.05),
                normalize,
                SpatialAugment(),
                TemporalSpeedPerturb(),
                derivatives,
                AddRandomTemporalShuffle(),
            ]
        ),
    )
    val_dataset = TrainLandmarkDataset(
        config.data_root,
        config.meta_path,
        config.val_driver_split,
        config.val_source_split,
        config.frames,
        Compose(
            [
                *common_pre,
                # No ClipCenter — evaluate() uses sliding window over full sequence
                normalize,
                derivatives,
            ]
        ),
    )
    sampler = SourceDriverCrossSampler(
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

def train(config: Settings, resume: Path | None = None) -> None:
    rank, world_size, local_rank = distributed_context()
    setup_logging(config.output_dir)
    seed_everything(config.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(config, rank, world_size)

    # Tracking setup
    csv_logger: CSVMetricsLogger | None = None
    if is_main_process():
        setup_mlflow_run("stgcn", "stgcn-train", config)
        csv_logger = CSVMetricsLogger(config.output_dir)

    raw_model = LandmarkSTGCN(
        make_anatomical_graph(config.vertices),
        config.channels,
        config.embedding_dim,
        config.embedding_dim,
        config.blocks,
        config.vertices,
        config.region_embed_dim,
        config.embed_bn,
        config.dropout,
        config.drop_path,
    ).to(device)

    model: nn.Module = raw_model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    criterion = FingerprintingLoss(config.temperature, config.graph_loss_weight).to(
        device
    )
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
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

    best_score = -math.inf
    patience_counter = 0
    loader_iterator = iter(train_loader)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Model parameters: %d total, %d trainable", total_params, trainable)

    for iteration in range(start + 1, config.iterations + 1):
        model.train()
        batch = next(loader_iterator)
        landmarks = (
            batch["landmarks_xyz"]
            .to(device, non_blocking=True)
            .reshape(-1, config.frames, config.vertices, config.channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        driving = batch["driving_id"].to(device, non_blocking=True).long()
        source = batch["source_id"].to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16,
        ):
            embedding, graph = model(landmarks)
        # Loss in float32 — LOGSOFT is numerically fragile in reduced precision
        loss, info = criterion(embedding.float(), graph.float(), driving, source)

        # Skip iteration if loss is non-finite (don't poison gradients/EMA)
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

        # Update EMA
        if ema is not None:
            ema.update(model)

        if is_main_process() and (iteration == 1 or iteration % 20 == 0):
            grad_str = f" | grad_norm {grad_norm:.4f}" if grad_norm is not None else ""
            LOGGER.info(
                "iteration %d/%d | loss %.6f | fingerprint %.6f | graph %.6f | lr %.2e%s",
                iteration,
                config.iterations,
                loss.item(),
                info["fingerprinting_loss"].item(),
                info["graph_loss"].item(),
                optimizer.param_groups[0]["lr"],
                grad_str,
            )
            train_metrics = {
                "loss": loss.item(),
                "fingerprint_loss": info["fingerprinting_loss"].item(),
                "graph_loss": info["graph_loss"].item(),
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
                    # Save the better model variant as best checkpoint
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
                        "New best model (%s) saved at iteration %d with score %.6f",
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

            # Broadcast early-stop decision to all workers
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
    model = LandmarkSTGCN(
        make_anatomical_graph(config.vertices),
        config.channels,
        config.embedding_dim,
        config.embedding_dim,
        config.blocks,
        config.vertices,
        config.region_embed_dim,
    ).to(device)
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
        ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
    )
    if distributed.is_initialized():
        distributed.destroy_process_group()


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
    # Hyperparameter overrides
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
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
    parser.add_argument(
        "--eval-distance", type=str, choices=["cosine", "euclidean"], default=None
    )
    args = parser.parse_args()

    # Build config with CLI overrides
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
    }
    for cli_attr, field_name in _cli_map.items():
        val = getattr(args, cli_attr, None)
        if val is not None:
            overrides[field_name] = val
    # Fields where CLI name matches Settings field name
    if getattr(args, "embed_bn", False):
        overrides["embed_bn"] = True
    for name in ("iterations", "seed", "dropout", "patience"):
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
