#!/usr/bin/env python3
"""Build a video+landmarks fusion submission, with a readable report.

Branches: our 6-model video ensemble (treated as one branch) plus any landmark models
(ST-GCN, NVIDIA-TCNN, KTDNN). For each it prints the individual score, runs greedy forward
selection (equal weights, as in Yehia's ensemble.py), then additionally sweeps the weight
simplex — greedy answers "which models", the sweep answers "in what proportion", and the two
usually disagree.

Fusion method: the per-model cosine SIMILARITY matrices are combined. At submission time the
same thing is achieved by L2-normalising each model's embedding, scaling by sqrt(weight/total)
and concatenating, so that the cosine of two concatenated vectors is exactly the weighted mean
of the per-model cosines. Validation and the submitted file therefore compute the same
quantity.

NOT done here, deliberately:
  * no centering of test embeddings by their own mean (that is a test-set statistic, and doing
    it per generation amounts to per-generator normalisation)
  * no averaging of raw embeddings across models — different models have unrelated coordinate
    systems, so that is not equivalent to averaging similarities

Usage:
    python scripts/build_fusion_submission.py --val-only        # report only
    python scripts/build_fusion_submission.py                   # report + test + zip
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import blosc2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODES = ("speaking", "listening")
VIDEO_MEMBERS = ["b1_batch128", "b3_batch128_seed1", "b5_batch128_seed2",
                 "gc3_b256_lr20e-3", "f2_featdiff_b128_6000it", "al1_aligned_s0"]
LANDMARKS = ["ktdnn_1", "ktdnn_2", "stgcn", "nvidia"]
BAR = "=" * 78
SUB = "-" * 78


def banner(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{SUB}\n", flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(m)
    finally:
        if str(path.parent) in sys.path:
            sys.path.remove(str(path.parent))
    return m


# --------------------------------------------------------------------- loaders
def ktdnn_embedder(tag: str, device, use_ema: bool):
    """Architecture is auto-detected from the state dict (as in Yehia's ensemble.py)."""
    d = ROOT / f"output/landmark_2d_106/{tag}"
    mod = load_module(f"ktdnn_trainer_{tag}_{int(use_ema)}", d / "trainer.py")
    ck = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
    key = "ema" if (use_ema and "ema" in ck and ck["ema"]) else "model"
    sd = dict(ck[key])
    in_ch = sd["tdnn.input_bn.weight"].shape[0]
    features = "raw" if in_ch == 636 else "kinematic"
    channels = sd["tdnn.stem.0.weight"].shape[0]
    if "tdnn.head.1.weight" in sd and sd["tdnn.head.1.weight"].ndim == 2:   # pre-dropout head
        remap = {}
        for k, v in sd.items():
            if k.startswith("tdnn.head.1."):
                remap[k.replace("tdnn.head.1.", "tdnn.head.2.")] = v
            elif k.startswith("tdnn.head.2."):
                remap[k.replace("tdnn.head.2.", "tdnn.head.3.")] = v
            else:
                remap[k] = v
        sd = remap
    model = mod._build_model(mod.Settings(features=features, tdnn_channels=channels), device)
    model.load_state_dict(sd)
    model.eval()

    def fn(raw: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            e = model(raw.unsqueeze(0).to(device).float())
        return F.normalize(e, p=2, dim=-1).squeeze(0).cpu()
    return fn, key


def window_model(tag: str, device, ckpt: Path | None = None):
    """ST-GCN / NVIDIA-TCNN, evaluated with whole-clip windows (default: shipped checkpoint)."""
    path = ROOT / ("models/landmarks/stgcn/trainer.py" if tag == "stgcn"
                   else "models/landmarks/nvidia/src/trainer.py")
    ckpt = ckpt or ROOT / ("output/landmark_2d_106/stgcn/best_model.pt" if tag == "stgcn"
                           else "output/landmark_2d_106/nvidia_tcnn/best_model.pt")
    st = load_module(f"{tag}_trainer", path)
    cfg = replace(st.Settings(), workers=0, frames=100, window_stride=36)
    if tag == "stgcn":
        model = st.LandmarkSTGCN(st.make_anatomical_graph(cfg.vertices), cfg.channels,
                                 cfg.embedding_dim, cfg.embedding_dim, cfg.blocks,
                                 cfg.vertices, cfg.region_embed_dim).to(device)
    else:
        model = st.PairwiseDistanceCNN(num_pairs=2 * cfg.num_pairs, embedding_dim=cfg.embedding_dim,
                                       stage_channels=cfg.stage_channels, stage_blocks=cfg.stage_blocks,
                                       stage_kernels=cfg.stage_kernels).to(device)
    st.load_checkpoint(ckpt, model)
    model.eval()
    return st, cfg, model


# --------------------------------------------------------------- validation data
def val_items():
    meta = json.loads((ROOT / "data/meta/train-landmark_2d_106-meta.json").read_text())
    dv = set(json.loads((ROOT / "data/train_val_splits/driver_val_splits.json").read_text()))
    sv = set(json.loads((ROOT / "data/train_val_splits/source_val_splits.json").read_text()))
    root = ROOT / "data/landmark_2d_106/train"
    return [(it, root / it["relpath"]) for it in meta
            if it["driver_id"] in dv and it["source_id"] in sv and (root / it["relpath"]).is_file()]


def clip_key(relpath: str) -> str:
    return relpath.replace("\\", "/").rsplit("/", 1)[0]


def video_branch(device, split: bool):
    """Per-model similarity matrices for the video models. If split=False they are merged into
    a single 'video' branch (equal weights), which is how the earlier reports treated them."""
    vt = load_module("mdnet_trainer", ROOT / "models/video/mdnet/trainer.py")
    vt.setup_logging(ROOT / "output/video/scratch_fusion_build")
    orig = vt.settings_from_checkpoint
    vt.settings_from_checkpoint = lambda ck: replace(orig(ck), frames=100, window_stride=36)
    vt.EVAL_TTA = 5
    per, info = {}, {}
    for name in VIDEO_MEMBERS:
        ck = torch.load(ROOT / f"output/video/study/{name}/best_model.pt", map_location="cpu", weights_only=False)
        cfg = vt.settings_from_checkpoint(ck)
        model = vt.build_model(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        ds = vt.CropDataset(cfg, cfg.val_driver_split, cfg.val_source_split, train=False)
        e = {}
        with torch.no_grad():
            for i in range(len(ds)):
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    v = vt.embed_clip(model, ds[i]["clip"], device, cfg.frames, cfg.window_stride)
                meta = ds.items[i]
                k = meta["relpath"][:-4]
                e[k] = F.normalize(v.float().cpu(), dim=-1)
                info[k] = (meta["set"], meta["mode"], meta["driver_id"])
        per[name] = e
        del model
        torch.cuda.empty_cache()
        print(f"  video model       : {name}", flush=True)
    keys = sorted(set.intersection(*[set(v) for v in per.values()]))
    mats = {n: (lambda M: M @ M.T)(torch.stack([per[n][k] for k in keys])) for n in VIDEO_MEMBERS}
    if split:
        return mats, keys, info
    merged = sum(m / len(VIDEO_MEMBERS) for m in mats.values())
    return {"video": merged}, keys, info


def make_scorer(keys, info):
    vt = sys.modules["mdnet_trainer"]
    drv = {d: i for i, d in enumerate(sorted({info[k][2] for k in keys}))}
    lab = torch.tensor([drv[info[k][2]] for k in keys])
    idx = {(s, m): torch.tensor([i for i, k in enumerate(keys) if info[k][0] == s and info[k][1] == m])
           for s in ("probe", "gallery") for m in MODES}

    def score(sim):
        p = {}
        for pm in MODES:
            for gm in MODES:
                a, b = idx[("probe", pm)], idx[("gallery", gm)]
                p[f"{pm}_vs_{gm}"] = vt.compute_auc(
                    sim[a][:, b].flatten(), (lab[a][:, None] == lab[b][None, :]).flatten().float())
        spk, lst = p["speaking_vs_speaking"], p["listening_vs_listening"]
        cross = (p["speaking_vs_listening"] + p["listening_vs_speaking"]) / 2
        return {"spk": spk, "lst": lst, "cross": cross,
                "score3": (spk + lst + cross) / 3,
                "score4": sum(p.values()) / 4, "pairs": p}
    return score


def greedy_forward_selection(names, sims, score):
    ranked = sorted(((n, score(sims[n])["score3"]) for n in names), key=lambda x: -x[1])
    best_name, best = ranked[0]
    selected, running = [best_name], sims[best_name].clone()
    print(f"  Start: {best_name:<16s} score={best:.6f}")
    for n, _ in ranked[1:]:
        cand = (running * len(selected) + sims[n]) / (len(selected) + 1)
        s = score(cand)["score3"]
        if s > best:
            print(f"  + {n:<16s} score={s:.6f}  ({s - best:+.6f}) ADDED")
            best, running, selected = s, cand, selected + [n]
        else:
            print(f"  - {n:<16s} score={s:.6f}  ({s - best:+.6f}) SKIPPED")
    return selected, best


def sweep(names, sims, score, step=0.05):
    """Exhaustive simplex sweep. Only tractable for a handful of branches."""
    grid = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    rows = []
    for combo in itertools.product(grid, repeat=len(names) - 1):
        w0 = round(1.0 - sum(combo), 2)
        if w0 < -1e-9:
            continue
        w = [w0] + list(combo)
        fused = sum(wi * sims[n] for wi, n in zip(w, names) if wi)
        r = score(fused)
        rows.append({"weights": dict(zip(names, w)), **{k: r[k] for k in ("score3", "score4", "spk", "lst", "cross")}})
    rows.sort(key=lambda r: -r["score3"])
    return rows


def coordinate_ascent(names, sims, score, step=0.05, rounds=8):
    """Weight search for many branches, where the full simplex is far too large to enumerate
    (10 branches at 0.05 would be ~10M points). Starts from equal weights and repeatedly
    re-optimises one branch at a time, renormalising, until no single change helps."""
    grid = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    w = {n: 1.0 / len(names) for n in names}

    def fuse(weights):
        tot = sum(weights.values())
        return sum((v / tot) * sims[n] for n, v in weights.items() if v > 0)

    best = score(fuse(w))["score3"]
    print(f"  start (equal weights)            score={best:.6f}")
    evals = len(names)
    for r in range(rounds):
        improved = False
        for n in names:
            cur = w[n]
            for cand in grid:
                if cand == cur:
                    continue
                trial = dict(w, **{n: cand})
                if sum(trial.values()) <= 0:
                    continue
                s = score(fuse(trial))["score3"]
                evals += 1
                if s > best + 1e-7:
                    best, w, cur, improved = s, trial, cand, True
        print(f"  round {r + 1}                          score={best:.6f}")
        if not improved:
            break
    tot = sum(w.values())
    w = {n: round(v / tot, 4) for n, v in w.items()}
    r = score(fuse(w))
    return {"weights": w, **{k: r[k] for k in ("score3", "score4", "spk", "lst", "cross")}}, evals


# ------------------------------------------------------------------ test embeddings
def test_embeddings(tag: str, geni: str, device, use_ema: bool) -> torch.Tensor:
    meta = json.loads((ROOT / f"data/meta/test_annon-landmark_2d_106-{geni}-meta.json").read_text())
    root = ROOT / f"data/landmark_2d_106/test-{geni}"
    fn, _ = ktdnn_embedder(tag, device, use_ema)
    out = []
    for i, item in enumerate(meta, 1):
        raw = blosc2.load_tensor(str(root / item["relpath"]))
        raw = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
        out.append(fn(raw))
        if i % 10000 == 0:
            print(f"    {tag} {geni}: {i}/{len(meta)}", flush=True)
    return F.normalize(torch.stack(out), p=2, dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-only", action="store_true", help="report only; skip test inference and zip")
    ap.add_argument("--merge-video", action="store_true",
                    help="treat the video models as a single 'video' branch instead of scoring each one")
    ap.add_argument("--out", default="submission/wba2026-fusion-video6-4landmark.zip")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(BAR)
    print("WBA2026 — video + landmarks fusion submission")
    print(BAR)
    print(f"  device            : {device}")
    print(f"  video models      : {len(VIDEO_MEMBERS)} "
          + ("merged into one branch" if a.merge_video else "scored as separate branches"))
    print(f"  landmark branches : {', '.join(LANDMARKS)}")

    banner("Embedding validation set")
    sims, keys, info = video_branch(device, split=not a.merge_video)
    score = make_scorer(keys, info)
    print(f"  validation clips  : {len(keys)}")
    items = val_items()
    ema_used = {}
    for tag in LANDMARKS:
        if tag.startswith("ktdnn"):
            best = None
            for use_ema in (False, True):
                fn, which = ktdnn_embedder(tag, device, use_ema)
                e = {clip_key(it["relpath"]): fn(torch.as_tensor(blosc2.load_tensor(str(p)))) for it, p in items}
                if any(k not in e for k in keys):
                    continue
                M = torch.stack([e[k] for k in keys])
                s = score(M @ M.T)["score3"]
                if best is None or s > best[0]:
                    best = (s, M @ M.T, which)
                if use_ema and "ema" not in torch.load(ROOT / f"output/landmark_2d_106/{tag}/best_model.pt",
                                                       map_location="cpu", weights_only=False):
                    break
            sims[tag], ema_used[tag] = best[1], best[2]
            print(f"  {tag:<18s}: weights={best[2]}  score={best[0]:.6f}")
        else:
            st, cfg, model = window_model(tag, device)
            _tr, loader = st.build_loaders(cfg, 0, 1)
            e = {}
            with torch.no_grad():
                for batch in loader:
                    out = st._sliding_window_embed(model, batch["landmarks_xyz"], device, cfg.frames, cfg.window_stride)
                    if isinstance(out, tuple):
                        out = out[0]
                    e["/".join(Path(batch["file"][0]).parts[-3:-1])] = F.normalize(out.float().cpu().reshape(-1), dim=-1)
            del model
            torch.cuda.empty_cache()
            M = torch.stack([e[k] for k in keys])
            sims[tag] = M @ M.T
            print(f"  {tag:<18s}: {len(e)} clips")

    names = list(sims)

    banner("Individual model scores (sorted)")
    ranked = sorted(((n, score(sims[n])) for n in names), key=lambda x: -x[1]["score3"])
    for i, (n, r) in enumerate(ranked, 1):
        print(f"  {i}. {n:<16s} spk={r['spk']:.6f}  lst={r['lst']:.6f}  "
              f"cross={r['cross']:.6f}  score={r['score3']:.6f}")

    banner("Greedy forward selection (equal weights)")
    selected, greedy_score = greedy_forward_selection(names, sims, score)

    print(f"\n{BAR}")
    print(f"SELECTED ENSEMBLE ({len(selected)} models): {', '.join(selected)}")
    print(f"ENSEMBLE SCORE: {greedy_score:.6f}")
    print(BAR)

    if len(names) <= 5:
        banner("Weight sweep (exhaustive, 0.05 resolution)")
        rows = sweep(names, sims, score)
        for r in rows[:8]:
            ws = "  ".join(f"{n}={r['weights'][n]:.2f}" for n in names)
            print(f"  {ws}  |  3-cond {r['score3']:.6f}  4-pair {r['score4']:.6f}")
        best = rows[0]
        near = sum(1 for r in rows if r["score3"] >= best["score3"] - 0.001)
        print(f"\n{len(rows)} combinations scored; {near} within 0.001 of the best")
    else:
        banner(f"Weight search — coordinate ascent ({len(names)} branches)")
        print(f"  the full simplex at 0.05 would be ~10M points, so one branch is re-optimised at a time\n")
        best, evals = coordinate_ascent(names, sims, score)
        print(f"\n{evals} evaluations")
        rows = [best]

    print(f"\n{BAR}")
    print("BEST WEIGHTED ENSEMBLE")
    print(SUB)
    for n in sorted(names, key=lambda x: -best["weights"][x]):
        bar = "#" * int(round(best["weights"][n] * 100))
        print(f"  {n:<26s} weight={best['weights'][n]:.4f}  {bar}")
    print(f"\n3-cond {best['score3']:.6f}   4-pair {best['score4']:.6f}")
    print(f"  spk={best['spk']:.6f}  lst={best['lst']:.6f}  cross={best['cross']:.6f}")
    print(f"  greedy equal-weight ensemble was {greedy_score:.6f}  "
          f"({best['score3'] - greedy_score:+.6f} from tuning the weights)")
    print(BAR)

    (ROOT / "output/fusion").mkdir(parents=True, exist_ok=True)
    (ROOT / "output/fusion/fusion_build_report.json").write_text(
        json.dumps({"individual": {n: score(sims[n]) for n in names},
                    "greedy": {"selected": selected, "score3": greedy_score},
                    "best_weighted": best, "ema_used": ema_used,
                    "top50": rows[:50]}, indent=1, default=float), encoding="utf-8")

    if a.val_only:
        print("\n--val-only: stopping before test inference.\n")
        return

    banner("Test inference (KTDNN branches)")
    build = ROOT / "output/video/submission_build"
    for tag in [t for t in LANDMARKS if t.startswith("ktdnn")]:
        for geni in ("gen1", "gen2"):
            out = build / f"{tag}-{geni}.bl2"
            if out.exists():
                print(f"  skip {tag} {geni} (exists)")
                continue
            emb = test_embeddings(tag, geni, device, ema_used.get(tag) == "ema")
            sys.path.insert(0, str(ROOT / "models/video/mdnet"))
            from infer import save_embeddings  # noqa: E402
            save_embeddings(emb, out)
            print(f"  wrote {out.name}  {tuple(emb.shape)}")

    banner("Done — combine with scripts/combine_video_embeddings.py using the weights above")
    print("  video branch file : output/video/submission_build/video6_aligned/embeddings-gen{1,2}.bl2")
    print("  stgcn             : output/landmark_2d_106/stgcn/submission/embeddings-gen{1,2}.bl2")
    print("  nvidia            : output/landmark_2d_106/nvidia_tcnn/submission/embeddings-gen{1,2}.bl2")
    print("  ktdnn_1 / ktdnn_2 : output/video/submission_build/ktdnn_{1,2}-gen{1,2}.bl2")
    print(f"  then package as   : {a.out}  (modality video+landmarks)\n")


if __name__ == "__main__":
    main()
