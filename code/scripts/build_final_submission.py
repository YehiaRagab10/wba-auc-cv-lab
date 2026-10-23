#!/usr/bin/env python3
"""Rebuild the FINAL WBA2026 submission (CodaBench 980274, 0.927 / 0.685 / 0.806) from the
per-branch test embeddings, and verify it against the uploaded zip.

gen1 file (same generator)  -- 15 branches, weights C51B5AL23:
    per-branch L2 norm -> per-avatar kNN centering (402 nearest clips in static face-shape
    space) -> per-branch removal of the mean of each predicted speaking/listening mode ->
    weighted concatenation -> DBA (k=8) applied twice.
gen2 file (different generator) -- 13 branches ("robust3nv"):
    12 correlation-descriptor branches (global per-generation mean centering) + the NVIDIA
    TCNN retrained at temperature 0.1 (per-avatar kNN centering) -> weighted concatenation ->
    DBA (k=8) applied twice.
Both files are zero-padded to a common width (cosine unchanged) and zipped with
scripts/prepare_submission_file.py.

Prerequisites (each documented in code_submission/README.md):
  per-branch test embeddings, output/fusion/avatar_shape/{train,test-gen1,test-gen2}.npy
  (scripts/avatar_shape_features.py), output/fusion/mode_pred/test-gen1_pspeak.npy
  (scripts/mode_classifier.py).

    .venv/Scripts/python.exe scripts/build_final_submission.py --out submission/final_rebuild.zip \
        --verify submission/wba2026-al23-gen2robust3nv.zip
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import blosc2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_avatar_centered import C51B5AL23, C65W, build, dba, shape_neighbours, voted_mode  # noqa: E402

AVATAR_K, DBA_K, DBA_ROUNDS = 402, 8, 2


def gen2_weights() -> dict[str, float]:
    """robust3nv: c65 without the two ECAPA-TDNN branches, + 3 robust correlation branches,
    then only correlation branches + the retrained NVIDIA (nv_t01_s1) in nvidia's place."""
    robust = {**{n: v for n, v in C65W.items() if not n.startswith("ktdnn")},
              "c_coherence": 0.0766, "c_rank_coherence": 0.0766, "c_band_mid": 0.0766}
    r3 = {n: v for n, v in robust.items() if n.startswith("c_") or n == "nvidia"}
    return {**{k: v for k, v in r3.items() if k != "nvidia"}, "nv_t01_s1": r3["nvidia"]}


def build_gen1() -> torch.Tensor:
    nn = shape_neighbours("gen1", AVATAR_K)
    e = build(C51B5AL23, "gen1", nn)
    mode = voted_mode(e, "gen1")
    e = build(C51B5AL23, "gen1", nn, mode=mode)
    for _ in range(DBA_ROUNDS):
        e = dba(e, DBA_K)
    return e


def build_gen2() -> torch.Tensor:
    nn = shape_neighbours("gen2", AVATAR_K)
    e = build(gen2_weights(), "gen2", nn, corr_global=True)
    for _ in range(DBA_ROUNDS):
        e = dba(e, DBA_K)
    return e


def load_zip(zp: Path, gen: str) -> torch.Tensor:
    with zipfile.ZipFile(zp) as z, tempfile.TemporaryDirectory() as td:
        p = Path(td) / "e.bl2"
        p.write_bytes(z.read(f"embeddings-{gen}.bl2"))
        return torch.as_tensor(blosc2.load_tensor(str(p))).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default=None, help="uploaded zip to compare against")
    a = ap.parse_args()
    g1, g2 = build_gen1(), build_gen2()
    w = max(g1.shape[1], g2.shape[1])
    pad = lambda t: torch.cat([t, torch.zeros(t.shape[0], w - t.shape[1])], 1)  # noqa: E731
    g1, g2 = pad(g1), pad(g2)
    work = ROOT / "output/fusion/_final_rebuild"
    work.mkdir(parents=True, exist_ok=True)
    for name, t in (("gen1", g1), ("gen2", g2)):
        assert t.shape[0] == 40200 and torch.isfinite(t).all() and float(t.norm(dim=1).min()) > 0.999
        blosc2.save_tensor(t.contiguous(), str(work / f"embeddings-{name}.bl2"), mode="w")
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_submission_file.py"),
                    str(work / "embeddings-gen1.bl2"), str(work / "embeddings-gen2.bl2"),
                    "cosine", "video+landmarks", "--out", a.out], check=True, capture_output=True)
    print(f"wrote {a.out}  width {w}")
    if a.verify:
        for name, t in (("gen1", g1), ("gen2", g2)):
            ref = load_zip(Path(a.verify), name)
            cs = F.cosine_similarity(F.normalize(ref, dim=-1), F.normalize(t, dim=-1), dim=-1)
            print(f"verify {name}: shape {tuple(ref.shape)} vs {tuple(t.shape)}  "
                  f"row cosine min {cs.min():.6f} mean {cs.mean():.6f}")


if __name__ == "__main__":
    main()
