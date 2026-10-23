#!/usr/bin/env python3
"""Test-set embeddings for the video branch (motion-differencing encoder).

Reads the cached face crops of a test generation (see preprocess.py), embeds every clip with a
trained checkpoint (sliding windows over the clip, averaged, L2-normalised) and writes a
``[N, D]`` blosc2 tensor in metadata order - the format scripts/prepare_submission_file.py expects.

Usage (project root, venv active):
    python models/video/mdnet/infer.py --geni gen1 --checkpoint output/video/mdnet_pixdiff/best_model.pt
    python models/video/mdnet/infer.py --geni gen2 --checkpoint output/video/mdnet_pixdiff/best_model.pt
Then:
    python scripts/prepare_submission_file.py <emb-gen1.bl2> <emb-gen2.bl2> cosine video --out <zip>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import trainer as T  # noqa: E402  (also puts models/ on sys.path)
from preprocess import load_crop  # noqa: E402

LOGGER = logging.getLogger("mdnet-infer")
EXPECTED = 40200


class TestCrops(Dataset):
    def __init__(self, cache_root: Path, meta_path: Path, limit: int | None) -> None:
        self.root = cache_root
        self.meta = json.loads(meta_path.read_text())
        keys = [m["relpath"] for m in self.meta]
        if len(keys) != EXPECTED or len(set(keys)) != EXPECTED:
            raise ValueError(f"Expected {EXPECTED} unique test videos in metadata")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if limit:
            self.meta = self.meta[:limit]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.root / (self.meta[index]["relpath"][:-4] + ".b2c")
        return torch.from_numpy(np.ascontiguousarray(load_crop(path)))


def save_embeddings(embeddings: torch.Tensor, path: Path) -> None:
    """Same Blosc2 layout as the landmark inference scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = embeddings.detach().cpu().float().contiguous()
    row_bytes = emb.shape[1] * emb.element_size()
    chunk = max(emb.element_size(), min(emb.nbytes, (32 * 1024**2 // row_bytes) * row_bytes))
    blosc2.save_tensor(
        emb, str(path), mode="w", chunksize=chunk,
        cparams={"codec": blosc2.Codec.ZSTD, "clevel": 3, "filters": [blosc2.Filter.BITSHUFFLE],
                 "typesize": emb.element_size(), "use_dict": False, "nthreads": os.cpu_count() or 1},
    )


def settings_from_checkpoint(ckpt: dict) -> T.Settings:
    return T.settings_from_checkpoint(ckpt)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geni", choices=["gen1", "gen2"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (testing)")
    ap.add_argument("--tta", type=int, default=1, help="test-time views per window (1 = off)")
    ap.add_argument("--frames", type=int, default=0, help="window length in frames (0 = training value; 100 = whole clip)")
    a = ap.parse_args()
    if a.limit and a.output is None:
        ap.error("--limit requires an explicit --output for the partial debug artifact")
    if not 1 <= a.tta <= len(T._VIEWS):
        ap.error("--tta must be between 1 and 6")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    T.EVAL_FRAMES = a.frames
    cfg = settings_from_checkpoint(ckpt)
    T.EVAL_TTA = a.tta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    LOGGER.info("loaded %s (variant=%s, iteration %s, weights=%s)", a.checkpoint, cfg.variant,
                ckpt.get("iteration"), ckpt.get("weights"))

    root = T.PROJECT_ROOT
    # the test cache must be built the same way as the training cache (e.g. 'train_96y' -> 'test-gen1_96y')
    suffix = "".join(ch for ch in Path(ckpt.get("config", {}).get("cache_root", "")).name.split("_")[-1] if ch.isalpha())
    ds = TestCrops(root / f"data/video_crops/test-{a.geni}_{cfg.image_size}{suffix}",
                   root / f"data/meta/test_annon-mp4-{a.geni}-meta.json", a.limit)
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                        persistent_workers=a.workers > 0)
    out: list[torch.Tensor] = []
    for i, clip in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            emb = T.embed_clip(model, clip, device, cfg.frames, cfg.window_stride)
        out.append(emb.float().cpu())
        if i % 5000 == 0:
            LOGGER.info("  %d/%d clips", i, len(ds))
    emb_all = torch.stack(out)
    norms = emb_all.norm(dim=1)
    LOGGER.info("embeddings %s | NaN: %s | norm min/max %.3f/%.3f", tuple(emb_all.shape),
                bool(torch.isnan(emb_all).any()), norms.min().item(), norms.max().item())
    if not a.limit and emb_all.shape[0] != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} embeddings, got {emb_all.shape[0]}")
    if not torch.isfinite(emb_all).all() or not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise RuntimeError("Invalid embeddings: require finite unit vectors for every video")
    path = a.output or a.checkpoint.parent / "submission" / f"embeddings-{a.geni}.bl2"
    save_embeddings(emb_all, path)
    path.with_suffix(".rows.json").write_text(json.dumps({
        "generation": a.geni, "complete": len(ds) == EXPECTED, "metric": "cosine",
        "rows": [m["relpath"] for m in ds.meta],
    }), encoding="utf-8")
    LOGGER.info("saved %s", path)


if __name__ == "__main__":
    main()
