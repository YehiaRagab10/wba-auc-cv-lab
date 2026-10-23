"""§1.1 One-time landmark cache.

Decodes every blosc2 landmark file once into three flat artefacts per split:

    cache/{split}_lm.f16.npy        [N_frames_total, 106, 2] float16, raw pixel coords
    cache/{split}_offsets.i64.npy   [N_seq + 1] start offsets into the above
    cache/{split}_meta.parquet      seq_idx, driver_id, source_id, mode, set, length, relpath

Raw pixel coordinates are stored with no normalisation baked in, so changing the
feature front-end never requires rebuilding the cache.

Sequence order is exactly the order of the official meta JSON, because submission
row order depends on it. Offsets are derived from the meta ``size`` field so that
workers can write into a preallocated memmap in parallel; a file whose real length
differs from ``size`` is truncated or end-padded to fit its slot, and the real
length is reported at the end.

Usage
-----
    python models/landmarks/dcs/build_cache.py --split train
    python models/landmarks/dcs/build_cache.py --split test-gen1
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache"
VERTICES = 106

META = {
    "train": "data/meta/train-landmark_2d_106-meta.json",
    "test-gen1": "data/meta/test_annon-landmark_2d_106-gen1-meta.json",
    "test-gen2": "data/meta/test_annon-landmark_2d_106-gen2-meta.json",
}


def _worker(args) -> list[tuple[int, int]]:
    """Decode a contiguous chunk of sequences into the memmap.

    Returns (seq_idx, real_length) for every sequence whose real length differed
    from the slot size it was given.
    """
    lm_path, lengths, offsets, jobs = args
    import blosc2  # imported per worker

    lm = np.memmap(lm_path, dtype=np.float16, mode="r+", shape=(offsets[-1], VERTICES, 2))
    mismatches: list[tuple[int, int]] = []
    for seq_idx, file_path in jobs:
        arr = np.asarray(blosc2.load_tensor(str(file_path)), dtype=np.float16)
        arr = arr.reshape(-1, VERTICES, 2)
        slot = int(lengths[seq_idx])
        real = arr.shape[0]
        if real != slot:
            mismatches.append((seq_idx, real))
            if real > slot:
                arr = arr[:slot]
            else:
                arr = np.concatenate([arr, np.repeat(arr[-1:], slot - real, axis=0)], axis=0)
        start = int(offsets[seq_idx])
        lm[start:start + slot] = arr
    lm.flush()
    del lm
    return mismatches


def build(split: str, workers: int, chunk: int) -> None:
    meta = json.loads((ROOT / META[split]).read_text())
    data_root = ROOT / "data/landmark_2d_106" / split

    lengths = np.array([int(it["size"]) for it in meta], dtype=np.int64)
    offsets = np.zeros(len(meta) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])

    CACHE.mkdir(parents=True, exist_ok=True)
    lm_path = CACHE / f"{split}_lm.f16.npy.raw"
    print(f"[{split}] {len(meta)} sequences, {total} frames, "
          f"{total * VERTICES * 2 * 2 / 2**30:.2f} GiB", flush=True)

    # Preallocate (sparse file on NTFS, so this is instant).
    with open(lm_path, "wb") as fh:
        fh.truncate(total * VERTICES * 2 * 2)

    jobs = [(i, data_root / it["relpath"]) for i, it in enumerate(meta)]
    missing = [p for _, p in jobs if not p.is_file()]
    if missing:
        raise SystemExit(f"[{split}] {len(missing)} files missing, first: {missing[0]}")

    batches = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    payload = [(str(lm_path), lengths, offsets, b) for b in batches]

    t0 = time.time()
    mismatches: list[tuple[int, int]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_worker, payload):
            mismatches.extend(res)
            done += 1
            if done % 20 == 0 or done == len(batches):
                el = time.time() - t0
                print(f"[{split}] {done}/{len(batches)} batches  {el:.0f}s  "
                      f"eta {el / done * (len(batches) - done):.0f}s", flush=True)

    # Wrap the raw buffer in a real .npy so np.load(mmap_mode="r") works.
    final = CACHE / f"{split}_lm.f16.npy"
    raw = np.memmap(lm_path, dtype=np.float16, mode="r", shape=(total, VERTICES, 2))
    with open(final, "wb") as fh:
        np.lib.format.write_array_header_2_0(
            fh, {"descr": "<f2", "fortran_order": False, "shape": (total, VERTICES, 2)}
        )
        step = 1_000_000
        for s in range(0, total, step):
            fh.write(np.ascontiguousarray(raw[s:s + step]).tobytes())
    del raw
    lm_path.unlink()

    np.save(CACHE / f"{split}_offsets.i64.npy", offsets)

    import pandas as pd
    df = pd.DataFrame({
        "seq_idx": np.arange(len(meta), dtype=np.int64),
        "relpath": [it["relpath"] for it in meta],
        "length": lengths,
        "driver_id": [it.get("driver_id", "") for it in meta],
        "source_id": [it.get("source_id", "") for it in meta],
        "mode": [it.get("mode", "") for it in meta],
        "set": [it.get("set", "") for it in meta],
    })
    df.to_parquet(CACHE / f"{split}_meta.parquet", index=False)

    print(f"[{split}] done in {time.time() - t0:.0f}s  -> {final.name} "
          f"({final.stat().st_size / 2**30:.2f} GiB)", flush=True)
    if mismatches:
        print(f"[{split}] WARNING {len(mismatches)} length mismatches vs meta size; "
              f"first 5: {mismatches[:5]}", flush=True)
    else:
        print(f"[{split}] all file lengths matched meta size", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=[*META, "all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    for split in (list(META) if a.split == "all" else [a.split]):
        build(split, a.workers, a.chunk)


if __name__ == "__main__":
    main()
