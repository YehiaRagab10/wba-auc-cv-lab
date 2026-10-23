#!/usr/bin/env python3
"""Build a GEOMETRICALLY ALIGNED crop cache: every frame is warped so the eyes, nose and mouth corners sit at a
canonical position, instead of the fixed per-clip square box used by preprocess.py.

Why: our models are strong at speaking-vs-speaking (0.92) but weak across behaviours (0.83-0.84). A model trained on
aligned crops sees facial *expression* motion with head pose removed, and the Aly branch's aligned model showed a much
smaller cross-mode gap (0.033 vs our 0.083). Alignment also normalises how a generator happens to frame the face, which
may transfer better across renderers. It does discard head motion, so the aligned model is intended as an ENSEMBLE
member alongside the existing unaligned ones, not as a replacement.

The per-frame transform (scale, rotation, tx, ty) is saved next to each clip as '<clip>.tf.npy'. That is exactly the
head motion the warp removes, so a later model can take it back as an auxiliary input without rebuilding this cache.

Deliberately NOT applied: the deflicker step from the Aly branch. Measured on 2026-09-23 it cost 0.016 of local
same-generator score with no gain on the robustness proxy (docs/video_ablation_study.md).

Usage (project root):
    python models/video/mdnet/preprocess_aligned.py --split train --size 96 --workers 6
    python models/video/mdnet/preprocess_aligned.py --split test-gen1 --size 96 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import blosc2
import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import LUMA_LUT, ROOT, SPLITS, SUFFIX, decode_gray, encode_crop  # noqa: E402

# 2D-106 landmark indices (same layout the landmark models use)
LEFT_EYE_IDX = list(range(33, 43))
RIGHT_EYE_IDX = list(range(87, 97))
NOSE_TIP_IDX = 72
LEFT_MOUTH_IDX = 52
RIGHT_MOUTH_IDX = 61
TARGET_IED_FRACTION = 0.4        # inter-eye distance as a fraction of the output side
TARGET_EYE_Y = 0.4               # eye line height as a fraction of the output side


class OneEuroFilter:
    """Low-pass filter whose cutoff rises with speed: removes landmark jitter without lagging real motion."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, 0.0
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        dx_hat = self._alpha(self.d_cutoff, dt) * dx + (1 - self._alpha(self.d_cutoff, dt)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


def smooth_points(points: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """One-Euro smoothing of [T, 5, 2] anchor trajectories, so the warp does not jitter frame to frame."""
    out = np.empty_like(points)
    for p in range(points.shape[1]):
        for a in range(points.shape[2]):
            filt = OneEuroFilter()
            for i, v in enumerate(np.nan_to_num(points[:, p, a], nan=0.0)):
                out[i, p, a] = filt(i / fps, float(v))
    return out


def extract_5_points(lm: np.ndarray) -> np.ndarray:
    """[T, 106, 2] -> [T, 5, 2]: eye centres, nose tip, mouth corners."""
    return np.stack([
        lm[:, LEFT_EYE_IDX].mean(axis=1),
        lm[:, RIGHT_EYE_IDX].mean(axis=1),
        lm[:, NOSE_TIP_IDX],
        lm[:, LEFT_MOUTH_IDX],
        lm[:, RIGHT_MOUTH_IDX],
    ], axis=1)


def canonical_points(out_size: int) -> np.ndarray:
    ied, eye_y = TARGET_IED_FRACTION, TARGET_EYE_Y
    return np.array([
        [0.5 - ied / 2, eye_y],
        [0.5 + ied / 2, eye_y],
        [0.5, eye_y + ied * 0.8],
        [0.5 - ied * 0.5, eye_y + ied * 1.2],
        [0.5 + ied * 0.5, eye_y + ied * 1.2],
    ], dtype=np.float32) * out_size


def align_frame(frame: np.ndarray, src: np.ndarray, dst: np.ndarray, out_size: int):
    """Similarity warp of one frame onto the canonical layout. Returns (crop, [scale, rotation, tx, ty])."""
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:                                    # degenerate landmarks: fall back to identity
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    crop = cv2.warpAffine(frame, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.arctan2(M[1, 0], M[0, 0]))
    return crop, np.array([scale, rot, float(M[0, 2]), float(M[1, 2])], dtype=np.float32)


def process(job) -> tuple[str, int, str | None]:
    relpath, mp4_root, lm_path, out_path, size, fast = job
    out = Path(out_path)
    if out.exists():
        return relpath, 0, None
    try:
        frames = decode_gray(Path(mp4_root) / relpath, fast)
        lm = np.asarray(blosc2.load_array(lm_path)).astype(np.float32)
        t = min(len(frames), len(lm))
        if t == 0:
            raise RuntimeError("no overlapping frames and landmarks")
        frames, lm = frames[:t], lm[:t]
        pts = smooth_points(extract_5_points(lm))
        dst = canonical_points(size)
        crops = np.empty((t, size, size), dtype=np.uint8)
        transforms = np.empty((t, 4), dtype=np.float32)
        for i in range(t):
            crops[i], transforms[i] = align_frame(frames[i], pts[i], dst, size)
        if fast:
            crops = LUMA_LUT[crops]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(encode_crop(crops))
        os.replace(tmp, out)
        np.save(out.with_suffix(".tf.npy"), transforms)   # head motion the warp removed, kept for later use
        return relpath, out.stat().st_size, None
    except Exception as exc:
        return relpath, 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/video_crops")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    ap.add_argument("--slow-decode", action="store_true")
    a = ap.parse_args()
    fast = not a.slow_decode
    cfg = SPLITS[a.split]
    out_dir = a.out_root / f"{a.split}_{a.size}{'y' if fast else ''}a"      # trailing 'a' = aligned
    meta = json.loads((ROOT / cfg["meta"]).read_text()) if isinstance(cfg["meta"], str) else json.loads(cfg["meta"].read_text())
    if a.limit:
        meta = meta[:a.limit]
    jobs = [(m["relpath"], str(cfg["mp4"]), str(cfg["lm"] / cfg["lm_rel"](m["relpath"])),
             str(out_dir / (m["relpath"][:-4] + SUFFIX)), a.size, fast) for m in meta]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.split}: {len(jobs)} clips -> {out_dir} | size {a.size} | workers {a.workers}", flush=True)
    t0, done, written, errors = time.time(), 0, 0, []
    with Pool(a.workers) as pool:
        for relpath, n, err in pool.imap_unordered(process, jobs, chunksize=8):
            done += 1
            written += n
            if err:
                errors.append(f"{relpath}\t{err}")
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} | {el/60:.1f} min | eta {(len(jobs)-done)*el/max(done,1)/60:.1f} min "
                      f"| {written/1e9:.2f} GB | {len(errors)} errors", flush=True)
    if errors:
        (out_dir / "errors.txt").write_text("\n".join(errors), encoding="utf-8")
    print(f"done: {done} clips, {written/1e9:.2f} GB, {len(errors)} errors"
          + (f" (see {out_dir/'errors.txt'})" if errors else ""), flush=True)


if __name__ == "__main__":
    main()
