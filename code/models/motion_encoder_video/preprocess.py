#!/usr/bin/env python3
"""Offline face-crop cache for the video branch.

Decoding the 512x512 FMP4 videos is the bottleneck (~220 ms/clip), so each clip is decoded
once, cropped around the face (box derived from the provided 2D-106 landmarks), converted to
grayscale, resized to SxS and stored losslessly compressed (temporal delta + zstd via blosc2).

File format (*.b2c):  3 x uint16 (T, S, S) header  +  blosc2 payload of the temporal-delta
uint8 array.  Use ``load_crop`` to read one back as a ``[T, S, S]`` uint8 array.

Usage (from the project root, in the venv):
    python models/video/mdnet/preprocess.py --split train --size 64 --workers 14
    python models/video/mdnet/preprocess.py --split test-gen1 --size 64 --workers 14
The run is resumable: clips whose output already exists are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import blosc2

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")  # raw-YUV decoding warns once per frame otherwise
import cv2
import numpy as np

cv2.setNumThreads(1)  # parallelism comes from processes; OpenCV threads do not speed up decoding
blosc2.set_nthreads(1)  # same for Blosc: one 16-thread pool per worker wastes memory and CPU

ROOT = Path(__file__).resolve().parents[3]
SPLITS = {
    "train": dict(
        mp4=ROOT / "data/mp4/train",
        lm=ROOT / "data/landmark_2d_106/train",
        meta=ROOT / "data/meta/train-mp4-meta.json",
        lm_rel=lambda r: r[:-4] + "/landmark_2d_106.bl2",
    ),
    "test-gen1": dict(
        mp4=ROOT / "data/mp4/test-gen1",
        lm=ROOT / "data/landmark_2d_106/test-gen1",
        meta=ROOT / "data/meta/test_annon-mp4-gen1-meta.json",
        lm_rel=lambda r: r[:-4] + ".bl2",
    ),
    "test-gen2": dict(
        mp4=ROOT / "data/mp4/test-gen2",
        lm=ROOT / "data/landmark_2d_106/test-gen2",
        meta=ROOT / "data/meta/test_annon-mp4-gen2-meta.json",
        lm_rel=lambda r: r[:-4] + ".bl2",
    ),
}
SUFFIX = ".b2c"
SIDE_MULT = 1.5   # crop side = SIDE_MULT * median landmark-box side (brows-to-chin -> adds forehead/cheek margin)
UP_SHIFT = 0.07   # shift the box up by this fraction of the side so the forehead is included
CPARAMS = blosc2.CParams(
    typesize=1, codec=blosc2.Codec.ZSTD, clevel=5, filters=[blosc2.Filter.SHUFFLE]
)


# Fast decoding hands over the video's own luma (Y) plane and skips the YUV->BGR->gray conversion
# (~2.7x faster). Y is limited-range (16..235); this LUT expands it to 0..255, which matches the
# BGR->gray path used for the original caches to within a fraction of a grey level (r = 1.0000).
LUMA_LUT = np.clip(np.round(1.164 * (np.arange(256) - 16)), 0, 255).astype(np.uint8)


def decode_gray(path: Path, fast: bool = True) -> np.ndarray:
    """Decode all frames to a [T, H, W] uint8 array. ``fast`` returns raw luma (needs LUMA_LUT)."""
    cap = cv2.VideoCapture(str(path))
    if fast:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:  # decoder returned colour anyway: convert the slow way
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif not fast:
            raise RuntimeError("unexpected single-channel frame in slow mode")
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded")
    return np.stack(frames)


def face_box(lm: np.ndarray) -> tuple[int, int, int]:
    """Fixed per-clip square box (x0, y0, side) from landmarks [T, 106, 2]."""
    w = lm[..., 0].max(1) - lm[..., 0].min(1)
    h = lm[..., 1].max(1) - lm[..., 1].min(1)
    side = SIDE_MULT * float(np.median(np.maximum(w, h)))
    cx = float(np.median((lm[..., 0].max(1) + lm[..., 0].min(1)) / 2))
    cy = float(np.median((lm[..., 1].max(1) + lm[..., 1].min(1)) / 2)) - UP_SHIFT * side
    side = max(int(round(side)), 8)
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def crop_resize(frames: np.ndarray, x0: int, y0: int, side: int, size: int) -> np.ndarray:
    _, h, w = frames.shape
    if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:  # box leaves the frame: pad with black
        frames = np.pad(frames, ((0, 0), (side, side), (side, side)))
        x0, y0 = x0 + side, y0 + side
    sub = frames[:, y0:y0 + side, x0:x0 + side]
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in sub])


def encode_crop(arr: np.ndarray) -> bytes:
    delta = arr.copy()
    delta[1:] = arr[1:] - arr[:-1]  # uint8 wraparound arithmetic is exactly invertible
    header = np.array(arr.shape, dtype=np.uint16).tobytes()
    return header + blosc2.compress2(delta.tobytes(), cparams=CPARAMS)


def load_crop(path: Path | str) -> np.ndarray:
    """Read a *.b2c file back into a [T, S, S] uint8 array."""
    raw = Path(path).read_bytes()
    t, s1, s2 = np.frombuffer(raw[:6], dtype=np.uint16)
    delta = np.frombuffer(blosc2.decompress2(raw[6:]), dtype=np.uint8).reshape(int(t), int(s1), int(s2))
    return np.cumsum(delta, axis=0, dtype=np.uint8)


def process(job: tuple[str, str, str, str, int, bool]) -> tuple[str, int, str | None]:
    """Returns (relpath, bytes_written, error). bytes_written == 0 and no error means skipped."""
    relpath, mp4_root, lm_path, out_path, size, fast = job
    out = Path(out_path)
    if out.exists():
        return relpath, 0, None
    try:
        frames = decode_gray(Path(mp4_root) / relpath, fast)
        lm = np.asarray(blosc2.load_array(lm_path)).astype(np.float32)
        x0, y0, side = face_box(lm)
        crops = crop_resize(frames, x0, y0, side, size)
        if fast:
            crops = LUMA_LUT[crops]
        payload = encode_crop(crops)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, out)
        return relpath, len(payload), None
    except Exception as exc:  # keep going; failures are listed at the end
        return relpath, 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips (benchmarking)")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/video_crops")
    parser.add_argument("--slow-decode", action="store_true",
                        help="original BGR->gray decoding (caches without the 'y' suffix, e.g. train_64)")
    args = parser.parse_args()

    fast = not args.slow_decode
    cfg = SPLITS[args.split]
    out_dir = args.out_root / f"{args.split}_{args.size}{'y' if fast else ''}"  # 'y' = fast luma decoding
    meta = json.loads(Path(cfg["meta"]).read_text())
    if args.limit:
        meta = meta[: args.limit]
    jobs = [
        (m["relpath"], str(cfg["mp4"]), str(cfg["lm"] / cfg["lm_rel"](m["relpath"])),
         str(out_dir / (m["relpath"][:-4] + SUFFIX)), args.size, fast)
        for m in meta
    ]
    print(f"{args.split}: {len(jobs)} clips -> {out_dir} | size {args.size} | workers {args.workers}", flush=True)

    t0 = time.time()
    done = written = 0
    errors: list[str] = []
    with Pool(args.workers) as pool:
        for relpath, nbytes, err in pool.imap_unordered(process, jobs, chunksize=4):
            done += 1
            written += nbytes
            if err:
                errors.append(f"{relpath}\t{err}")
            if done % 500 == 0 or done == len(jobs):
                rate = done / (time.time() - t0)
                eta = (len(jobs) - done) / rate / 60 if rate else float("inf")
                print(f"  {done}/{len(jobs)} | {rate:.1f} clips/s | ETA {eta:.0f} min | written {written/1e9:.2f} GB "
                      f"| errors {len(errors)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if errors:
        (out_dir / "errors.txt").write_text("\n".join(errors))
    print(f"finished in {(time.time()-t0)/60:.1f} min | errors {len(errors)}"
          + (f" (see {out_dir/'errors.txt'})" if errors else ""), flush=True)


if __name__ == "__main__":
    main()
