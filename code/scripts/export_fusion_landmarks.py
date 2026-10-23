"""Export ST-GCN/NVIDIA using the exact 100-frame/36-stride fusion evaluator.

Processes original landmark files (no fp16 cache), batches equal-length clips,
and keeps metadata row order. Normalization is per clip only. Existing baseline
exports are preserved; use a new output directory for this corrected recipe.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import blosc2
import torch
import torch.nn.functional as F

import build_fusion_submission as bf


@torch.inference_mode()
def embed_batch(module, cfg, model, features, device):
    out = module._sliding_window_embed(model, torch.stack(features), device,
                                       cfg.frames, cfg.window_stride)
    if isinstance(out, tuple):
        out = out[0]
    return F.normalize(out.float().cpu(), dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["stgcn", "nvidia"])
    ap.add_argument("--split", required=True, choices=["val", "gen1", "gen2"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--io-threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke check; explicitly marked partial")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="non-default checkpoint (same architecture); default = the shipped one")
    ap.add_argument("--name", default=None,
                    help="with --split val and --checkpoint: write val_sims/<name>-<hash>.pt "
                         "instead of asserting parity against the shipped branch")
    a = ap.parse_args()
    if a.batch < 1 or a.io_threads < 1 or a.limit < 0:
        ap.error("batch and io-threads must be positive; limit must be nonnegative")
    suffix = "-partial" if a.limit else ""
    dest = a.out_dir / f"embeddings-{a.split}{suffix}.bl2"
    manifest = dest.with_suffix(".rows.json")
    if dest.exists() or manifest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module, cfg, model = bf.window_model(a.model, device, a.checkpoint)
    _, val_loader = module.build_loaders(cfg, 0, 1)
    transform = val_loader.dataset.transform
    if a.split == "val":
        items = bf.val_items()
    else:
        meta_path = bf.ROOT / f"data/meta/test_annon-landmark_2d_106-{a.split}-meta.json"
        meta = json.loads(meta_path.read_text())
        if len(meta) != 40200:
            raise ValueError("Expected 40200 test metadata rows")
        items = [(it, bf.ROOT / f"data/landmark_2d_106/test-{a.split}" / it["relpath"]) for it in meta]
    if a.limit:
        items = items[:a.limit]
    keys = [bf.clip_key(it["relpath"]) for it, _ in items]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Expected nonempty, unique metadata rows")

    def prepare(path):
        return transform(torch.as_tensor(blosc2.load_tensor(str(path))).float())

    result = torch.empty(len(items), cfg.embedding_dim)
    started = time.perf_counter()
    parity_checked = False
    with ThreadPoolExecutor(max_workers=a.io_threads) as pool:
        for start in range(0, len(items), a.batch):
            features = list(pool.map(prepare, [p for _, p in items[start:start + a.batch]]))
            buckets = {}
            for local, f in enumerate(features):
                buckets.setdefault(tuple(f.shape), []).append(local)
            for indices in buckets.values():
                batch = [features[i] for i in indices]
                out = embed_batch(module, cfg, model, batch, device)
                if not parity_checked and len(batch) > 1:
                    single = embed_batch(module, cfg, model, batch[:1], device)
                    torch.testing.assert_close(out[:1], single, atol=2e-5, rtol=2e-4)
                    parity_checked = True
                result[[start + i for i in indices]] = out
            done = min(start + a.batch, len(items))
            if start == 0 or done % 512 == 0 or done == len(items):
                elapsed = time.perf_counter() - started
                print(f"{a.model} {a.split}: {done}/{len(items)}; "
                      f"{elapsed:.1f}s; ETA {elapsed / done * (len(items) - done):.0f}s", flush=True)
    if not torch.isfinite(result).all() or not torch.allclose(result.norm(dim=1), torch.ones(len(result)), atol=1e-5):
        raise ValueError("Nonfinite or nonunit output")
    checkpoint = a.checkpoint or bf.ROOT / "output/landmark_2d_106" / (
        "stgcn" if a.model == "stgcn" else "nvidia_tcnn") / "best_model.pt"
    record = {"generation": a.split, "complete": not bool(a.limit), "rows": keys,
              "model": a.model, "frames": cfg.frames, "stride": cfg.window_stride,
              "normalization": "per-clip L2, no population centering",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "batch_single_parity_checked": parity_checked}
    if a.split == "val" and not a.limit:
        blob = torch.load(bf.ROOT / "output/fusion/val_sims/video_keys_info.pt", weights_only=False)
        positions = {k: i for i, k in enumerate(keys)}
        aligned = result[[positions[k] for k in blob["keys"]]]
        digest = hashlib.sha1("\n".join(blob["keys"]).encode()).hexdigest()[:12]
        actual = aligned @ aligned.T
        if a.name:
            torch.save(actual, bf.ROOT / f"output/fusion/val_sims/{a.name}-{digest}.pt")
            print(f"Wrote val_sims/{a.name}-{digest}.pt", flush=True)
            return
        expected = torch.load(bf.ROOT / f"output/fusion/val_sims/{a.model}-{digest}.pt", weights_only=True)
        record["cached_similarity_max_abs_error"] = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        print(f"Full validation parity: max error {record['cached_similarity_max_abs_error']:.8f}", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    blosc2.save_tensor(result.contiguous(), str(dest), mode="w")
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
