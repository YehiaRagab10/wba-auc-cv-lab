#!/usr/bin/env python3
"""Assemble a working directory from this package so every command in the Reproduction Guide
runs unchanged.

The code locates its project root from its own position in the folder tree (as it did during the
competition), so the clean package layout (code/models/<model>, checkpoints/<model>/...) is mapped
back to the layout the code expects:

    <out>/wba/                       project root (models/, scripts/, output/, data/ -> your data)
    <out>/wba-video-best-0.8978/     video checkpoints and the folder for their test embeddings

Usage (from the package root):
    python code/scripts/setup_workspace.py --out D:/wba_workspace --data D:/WBA2026_data
then run the Reproduction Guide commands inside D:/wba_workspace/wba.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]

CODE = {                      # package code folder -> project-relative folder
    "code/models/common": "models",
    "code/models/correlation_descriptors": "models/landmarks/dcs",
    "code/models/motion_encoder_video": "models/video/mdnet",
    "code/models/stgcn": "models/landmarks/stgcn",
    "code/models/pairwise_distance_tcnn": "models/landmarks/nvidia/src",
    "code/models/ecapa_tdnn": ["output/landmark_2d_106/ktdnn_1", "output/landmark_2d_106/ktdnn_2"],
    "code/scripts": "scripts",
}
CKPT = {                      # package checkpoint -> project-relative path ("../" = next to wba/)
    "checkpoints/correlation_descriptors/c_d8_hand/best_model.pt": "output/landmarks/dcs/d8_hand_seed1/best_model.pt",
    "checkpoints/motion_encoder_video/b3/best_model.pt": "../wba-video-best-0.8978/checkpoints/b3_batch128_seed1/best_model.pt",
    "checkpoints/motion_encoder_video/b5/best_model.pt": "../wba-video-best-0.8978/checkpoints/b5_batch128_seed2/best_model.pt",
    "checkpoints/motion_encoder_video/al1/best_model.pt": "../wba-video-best-0.8978/checkpoints/al1_aligned_s0/best_model.pt",
    "checkpoints/motion_encoder_video/al2_aligned_s1/best_model.pt": "output/video/study/al2_aligned_s1/best_model.pt",
    "checkpoints/motion_encoder_video/al3_aligned_s2/best_model.pt": "output/video/study/al3_aligned_s2/best_model.pt",
    "checkpoints/stgcn/stgcn/best_model.pt": "output/landmark_2d_106/stgcn/best_model.pt",
    "checkpoints/pairwise_distance_tcnn/nvidia/best_model.pt": "output/landmark_2d_106/nvidia_tcnn/best_model.pt",
    "checkpoints/pairwise_distance_tcnn/nv_t01_s1/best_model.pt": "output/landmark_2d_106/variants/nv_t01_s1/best_model.pt",
    "checkpoints/ecapa_tdnn/ktdnn_1/best_model.pt": "output/landmark_2d_106/ktdnn_1/best_model.pt",
    "checkpoints/ecapa_tdnn/ktdnn_2/best_model.pt": "output/landmark_2d_106/ktdnn_2/best_model.pt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="new working directory")
    ap.add_argument("--data", type=Path, default=None,
                    help="folder holding the challenge data (meta/, landmark_2d_106/, mp4/, train_val_splits/)")
    a = ap.parse_args()
    root = a.out / "wba"
    root.mkdir(parents=True, exist_ok=True)
    (a.out / "wba-video-best-0.8978" / "embeddings").mkdir(parents=True, exist_ok=True)
    for src, dsts in CODE.items():
        for dst in ([dsts] if isinstance(dsts, str) else dsts):
            (root / dst).mkdir(parents=True, exist_ok=True)
            for f in (PKG / src).iterdir():
                if f.is_file():
                    shutil.copy2(f, root / dst / f.name)
    shutil.copy2(PKG / "code/requirements.txt", root / "requirements.txt")
    for src, dst in CKPT.items():
        target = (root / dst).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PKG / src, target)
    if a.data:
        data = root / "data"
        if not data.exists():
            shutil.copytree(a.data, data)
    print(f"workspace ready: {root}")


if __name__ == "__main__":
    main()
