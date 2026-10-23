#!/usr/bin/env python3
"""Fit PCA basis on training landmark deformations for the Kinematic-TDNN front-end.

Samples ~200k random frames from the training split, computes Procrustes-aligned
deformations (D_t = aligned_t - median(aligned)), and fits a PCA basis on the
flattened 212-dim vectors.

Saves the basis and mean as a .pt file for use by KinematicFrontEnd.

Usage:
    python fit_pca.py
    python fit_pca.py --n-frames 200000 --pca-k 32 --output pca_basis.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import blosc2
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
LOGGER = logging.getLogger("fit_pca")

LEFT_EYE_IDX = list(range(33, 43))
RIGHT_EYE_IDX = list(range(87, 97))


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def clip_normalise(x: torch.Tensor) -> torch.Tensor:
    """Subtract clip-mean centroid, divide by clip-median IOD.

    Args:
        x: [T, 106, 2]

    Returns:
        [T, 106, 2]
    """
    x = x.float()
    centroid = x.mean(dim=(0, 1), keepdim=True)
    x = x - centroid
    left_c = x[:, LEFT_EYE_IDX, :].mean(dim=1)
    right_c = x[:, RIGHT_EYE_IDX, :].mean(dim=1)
    iod = (left_c - right_c).norm(dim=-1).median()
    iod = iod.clamp_min(1e-6)
    return x / iod


def procrustes_align(x: torch.Tensor, n_iter: int = 2) -> torch.Tensor:
    """Per-frame Procrustes alignment to clip-mean shape.

    Args:
        x: [T, 106, 2]

    Returns:
        aligned: [T, 106, 2]
    """
    z = torch.complex(x[..., 0], x[..., 1])
    ref = z.mean(dim=0, keepdim=True)

    for _ in range(n_iter):
        numerator = (ref.conj() * z).sum(dim=-1)
        denominator = (z * z.conj()).real.sum(dim=-1)
        a_t = numerator / denominator.clamp_min(1e-10)
        aligned_z = a_t.unsqueeze(-1) * z
        ref = aligned_z.mean(dim=0, keepdim=True)

    return torch.stack([aligned_z.real, aligned_z.imag], dim=-1)


def compute_deformation(x: torch.Tensor) -> torch.Tensor:
    """Compute deformation: D_t = aligned_t - median(aligned).

    Args:
        x: [T, 106, 2] raw landmarks

    Returns:
        [T, 212] flattened deformation
    """
    normed = clip_normalise(x)
    aligned = procrustes_align(normed)
    median_shape = aligned.median(dim=0, keepdim=True).values
    deformation = aligned - median_shape
    return deformation.reshape(deformation.shape[0], -1)  # [T, 212]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=PROJECT_ROOT / "data/landmark_2d_106/train",
    )
    parser.add_argument(
        "--meta-path", type=Path,
        default=PROJECT_ROOT / "data/meta/train-landmark_2d_106-meta.json",
    )
    parser.add_argument(
        "--driver-split", type=Path,
        default=PROJECT_ROOT / "data/train_val_splits/driver_train_splits.json",
    )
    parser.add_argument(
        "--source-split", type=Path,
        default=PROJECT_ROOT / "data/train_val_splits/source_train_splits.json",
    )
    parser.add_argument("--n-frames", type=int, default=200_000,
                        help="Target total frames (sampled as ~10 per clip from ~20k clips)")
    parser.add_argument("--frames-per-clip", type=int, default=10)
    parser.add_argument("--max-clips", type=int, default=20_000)
    parser.add_argument("--pca-k", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=SCRIPT_DIR / "pca_basis.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load metadata and filter by train split
    driver_ids = set(json.loads(args.driver_split.read_text()))
    source_ids = set(json.loads(args.source_split.read_text()))
    metadata = json.loads(args.meta_path.read_text())
    meta = [
        item for item in metadata
        if item["driver_id"] in driver_ids
        and item["source_id"] in source_ids
        and (args.data_root / item["relpath"]).is_file()
    ]
    LOGGER.info("Found %d train sequences", len(meta))

    # Sample ~frames_per_clip random frames from each of ~max_clips clips
    # This ensures diversity across clips rather than taking all frames from a few
    random.shuffle(meta)
    clips_to_use = meta[:args.max_clips]
    all_deformations: list[torch.Tensor] = []
    total_frames = 0

    for item in tqdm(clips_to_use, desc="Processing sequences"):
        if total_frames >= args.n_frames:
            break
        try:
            landmarks = blosc2.load_tensor(args.data_root / item["relpath"])
            if not isinstance(landmarks, torch.Tensor):
                landmarks = torch.as_tensor(landmarks)
            if landmarks.shape[0] < 10:
                continue
            deformation = compute_deformation(landmarks)
            # Take a fixed small number of random frames per clip
            n_take = min(args.frames_per_clip, deformation.shape[0])
            idx = torch.randperm(deformation.shape[0])[:n_take]
            deformation = deformation[idx]
            all_deformations.append(deformation)
            total_frames += deformation.shape[0]
        except Exception as e:
            LOGGER.warning("Skipping %s: %s", item["relpath"], e)
            continue

    LOGGER.info("Collected %d frames for PCA", total_frames)

    # Fit PCA
    data = torch.cat(all_deformations, dim=0).float()  # [N, 212]
    mean = data.mean(dim=0)  # [212]
    centered = data - mean

    LOGGER.info("Computing SVD on [%d, 212] matrix...", centered.shape[0])
    # Use randomized SVD for speed if data is large
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    basis = Vt[:args.pca_k]  # [K, 212]

    # Variance explained
    total_var = S.square().sum()
    explained = S[:args.pca_k].square().sum() / total_var
    LOGGER.info(
        "PCA: %d components explain %.2f%% of variance",
        args.pca_k, 100 * explained.item(),
    )

    # Save
    torch.save({"basis": basis, "mean": mean, "singular_values": S}, args.output)
    LOGGER.info("Saved PCA basis to %s", args.output)


if __name__ == "__main__":
    main()