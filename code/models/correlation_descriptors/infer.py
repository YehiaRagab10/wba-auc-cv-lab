"""Test-set inference for Approach C.

Embeds a cached test split whole-sequence, in exactly the order of the official
meta JSON, and runs the same ``postprocess`` the validation path uses, so the two
cannot drift apart (plan §1.6 note 4).

Submission sanity checks, kept from the existing landmark ``infer.py``: one row per
meta entry, meta order preserved, float32, all values finite.

Usage
-----
    python models/landmarks/dcs/infer.py --checkpoint output/landmarks/dcs/c1_full/best_model.pt \\
        --split test-gen1 --out output/landmarks/dcs/c1_full/gen1.bl2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import LandmarkCache                                           # noqa: E402
from evaluate import PostStats, embed_split, postprocess                 # noqa: E402
from geometry import GeometricFrontEnd                                    # noqa: E402
from model import DCSNet                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def load(checkpoint: Path, device: torch.device, use_ema: bool = False):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, festate = ck["config"], ck.get("front_end", {})
    fe = GeometricFrontEnd(derivatives=festate.get("derivatives", True),
                           median_shape=festate.get("median_shape", True),
                           pose=festate.get("pose", True)).to(device)
    # Head geometry is read from the checkpoint, falling back to the original
    # (1024, 512) layout for checkpoints saved before the head became configurable.
    net = DCSNet(fe.out_channels, fe.handcrafted_channels(), learned=cfg["learned"],
                 kernel=cfg["kernel"], embedding_dim=cfg["embedding_dim"],
                 hidden=tuple(cfg.get("hidden", (1024, 512))),
                 dropout=cfg["dropout"], input_dropout=cfg.get("input_dropout", 0.0),
                 lags=tuple(cfg["lags"]),
                 use_handcrafted=cfg["use_handcrafted"], use_learned=cfg["use_learned"],
                 use_lags=cfg["use_lags"], use_amplitude=cfg["use_amplitude"]).to(device)
    state = ck["ema"] if use_ema and "ema" in ck else ck["model"]
    net.load_state_dict({k: v.to(net.state_dict()[k].dtype) for k, v in state.items()})
    net.eval()
    print(f"loaded {checkpoint.name} (ema={use_ema}) metrics={ck.get('metrics', {}).get('score3')}",
          flush=True)
    return fe, net, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, choices=["test-gen1", "test-gen2", "train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--centre", action="store_true",
                    help="subtract the TRAIN embedding mean (fitted here, never the test mean)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe, net, _ = load(Path(a.checkpoint), device, a.ema)

    stats = None
    if a.centre:
        tr_cache = LandmarkCache("train")
        from data import train_val_indices
        tr, _ = train_val_indices(tr_cache)
        sub = tr[torch.randperm(len(tr))[:8000]]
        e, _ = embed_split(net, fe, tr_cache, sub, device, a.batch)
        stats = PostStats(mean=e.mean(dim=0, keepdim=True))
        print(f"fitted train-mean centering on {len(sub)} clips", flush=True)
        del tr_cache

    cache = LandmarkCache(a.split)
    t0 = time.time()
    emb, order = embed_split(net, fe, cache, torch.arange(len(cache)), device, a.batch)

    # Restore official meta order: bucketing by length reorders the forward passes.
    emb = postprocess(emb, stats)
    restored = torch.empty_like(emb)
    restored[order] = emb
    print(f"embedded {len(restored)} clips in {time.time()-t0:.0f}s, dim {restored.shape[1]}",
          flush=True)

    assert len(restored) == len(cache.meta), f"{len(restored)} rows vs {len(cache.meta)} meta entries"
    assert torch.isfinite(restored).all(), "non-finite values in embeddings"
    out = restored.float().contiguous()

    import blosc2
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blosc2.save_tensor(out, str(dest), mode="w")
    print(f"wrote {dest} {tuple(out.shape)} float32", flush=True)


if __name__ == "__main__":
    main()
