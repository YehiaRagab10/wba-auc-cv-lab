#!/usr/bin/env python3
"""Inference script for Kinematic-TDNN embeddings.

Extracts embeddings from test sequences using a trained checkpoint.
Whole-sequence forward passes (no sliding windows needed).

Examples:
    python infer.py --geni gen1 --checkpoint output/landmark_2d_106/ktdnn/best_model.pt
    python infer.py --geni gen2 --checkpoint output/landmark_2d_106/ktdnn/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trainer import (
    KTDNNWithFrontEnd,
    PadShortClips,
    _build_model,
    load_checkpoint,
    postprocess,
    seed_everything,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
LOGGER = logging.getLogger("ktdnn-infer")


@dataclass(frozen=True)
class Settings:
    data_root: Path = PROJECT_ROOT / "data/landmark_2d_106/test"
    meta_path: Path = PROJECT_ROOT / "data/meta/test_annon-landmark_2d_106-gen1-meta.json"
    output_dir: Path = PROJECT_ROOT / "output/landmark_2d_106/ktdnn/submission"
    output_embeddings_name: str = "embeddings.bl2"
    features: str = "kinematic"
    pca_path: Path | None = None
    pca_k: int = 32
    frames: int = 75
    min_crop: int = 48
    max_crop: int = 160
    vertices: int = 106
    channels: int = 303
    tdnn_channels: int = 384
    embedding_dim: int = 256
    window_stride: int = 37
    workers: int = 4
    seed: int = 0

    @classmethod
    def from_geni(cls, geni: str) -> Settings:
        if geni not in ["gen1", "gen2"]:
            raise ValueError(f"Invalid geni: {geni}. Must be 'gen1' or 'gen2'.")
        return cls(
            data_root=PROJECT_ROOT / f"data/landmark_2d_106/test-{geni}",
            meta_path=PROJECT_ROOT / f"data/meta/test_annon-landmark_2d_106-{geni}-meta.json",
            output_embeddings_name=f"embeddings-{geni}.bl2",
        )


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "infer.log"),
        ],
        force=True,
    )


class InferenceLandmarkDataset(Dataset):
    """Blosc2 landmark dataset for inference."""

    def __init__(
        self,
        root: Path,
        meta_path: Path,
        min_frames: int = 48,
    ) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"Landmark root does not exist: {root}")
        metadata = json.loads(meta_path.read_text())
        self.root = root
        self.meta = metadata
        self.pad = PadShortClips(min_frames)
        if not self.meta:
            raise RuntimeError(f"No usable samples {meta_path}.")
        LOGGER.info("Loaded %d samples from %s", len(self.meta), meta_path)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.meta[index]
        landmarks = blosc2.load_tensor(self.root / item["relpath"])
        if not isinstance(landmarks, Tensor):
            landmarks = torch.as_tensor(landmarks)
        landmarks = self.pad(landmarks)
        return {
            "landmarks_xyz": landmarks,
            "file": (self.root / item["relpath"]).as_posix(),
        }


def save_embeddings(
    embeddings: torch.Tensor,
    path: str | Path,
    *,
    target_chunk_mb: int = 32,
    compression_level: int = 3,
    threads: int | None = None,
    overwrite: bool = True,
) -> int:
    """Save embedding matrix using Blosc2."""
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    if target_chunk_mb <= 0:
        raise ValueError("target_chunk_mb must be greater than zero")

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = embeddings.detach().cpu().contiguous()
    if embeddings.dim() == 0:
        raise ValueError("embeddings must have at least one dimension")

    target_chunk_bytes = target_chunk_mb * 1024**2
    if embeddings.ndim >= 2:
        row_bytes = int(np.prod(embeddings.shape[1:])) * embeddings.dtype.itemsize
        rows_per_chunk = max(1, target_chunk_bytes // row_bytes)
        chunksize = min(embeddings.nbytes, rows_per_chunk * row_bytes)
    else:
        chunksize = min(embeddings.nbytes, target_chunk_bytes)
    chunksize = max(embeddings.dtype.itemsize, chunksize)

    return blosc2.save_tensor(
        embeddings,
        str(output_path),
        mode="w",
        chunksize=chunksize,
        cparams={
            "codec": blosc2.Codec.ZSTD,
            "clevel": compression_level,
            "filters": [blosc2.Filter.BITSHUFFLE],
            "typesize": embeddings.dtype.itemsize,
            "use_dict": False,
            "nthreads": threads or os.cpu_count() or 1,
        },
    )


@torch.no_grad()
def run_inference(config: Settings, checkpoint: Path) -> None:
    setup_logging(config.output_dir)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = InferenceLandmarkDataset(
        config.data_root, config.meta_path, config.min_crop,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )

    # Build model using trainer's builder
    # We need a trainer-compatible config for _build_model
    from trainer import Settings as TrainerSettings
    trainer_config = TrainerSettings(
        features=config.features,
        pca_path=config.pca_path,
        pca_k=config.pca_k,
        tdnn_channels=config.tdnn_channels,
        embedding_dim=config.embedding_dim,
    )
    model = _build_model(trainer_config, device)
    load_checkpoint(checkpoint, model)
    model.eval()

    embeddings: list[Tensor] = []
    for batch in tqdm(loader, desc="Inference"):
        raw = batch["landmarks_xyz"]  # [1, T, V, C]
        x = raw.to(device, non_blocking=True).float()

        # Whole-sequence forward
        emb = model(x)  # [1, D]
        emb = F.normalize(emb, p=2, dim=-1)
        embeddings.append(emb.cpu())

    result = torch.cat(embeddings, dim=0)

    # Post-process (centering + re-normalise) — same as evaluate()
    result = postprocess(result)

    LOGGER.info(
        "Embedding count: %d | shape: %s (expected 40200 per generation)",
        result.shape[0], tuple(result.shape),
    )
    if result.shape[0] != 40200:
        LOGGER.warning(
            "Expected 40200 embeddings but got %d — submission may be invalid!",
            result.shape[0],
        )
    output_path = config.output_dir / config.output_embeddings_name
    save_embeddings(result, output_path, overwrite=True)
    LOGGER.info("Saved %d embeddings to %s", result.shape[0], output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geni", type=str, choices=["gen1", "gen2"], default="gen1",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "output/landmark_2d_106/ktdnn/best_model.pt",
    )
    parser.add_argument("--features", type=str, choices=["kinematic", "raw"], default=None)
    parser.add_argument("--pca-path", type=Path, default=None)
    args = parser.parse_args()

    config = Settings.from_geni(args.geni)
    overrides: dict[str, Any] = {}
    if args.features is not None:
        overrides["features"] = args.features
        overrides["channels"] = 636 if args.features == "raw" else 303
    if args.pca_path is not None:
        overrides["pca_path"] = args.pca_path
    if overrides:
        config = replace(config, **overrides)

    run_inference(config, args.checkpoint)


if __name__ == "__main__":
    main()