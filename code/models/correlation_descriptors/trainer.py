"""Approach C (DCS) trainer — plan §4 on top of the §1 shared infrastructure.

Pipeline per step, all on GPU, no DataLoader:

    §1.5 sampler -> §1.2 cache gather -> §1.4 augment -> §1.3 front-end (raw, no PCA)
    -> §4 correlation signature -> embedding -> AAM + contrastive + source-GRL

Correlations need roughly 64 frames to be stable, so training crops are drawn from
[96, 200] frames and evaluation uses whole sequences (§1.6).

Every ablation in plan §4 is a CLI flag: ``--no-handcrafted``, ``--no-learned``,
``--no-lags``, ``--no-amplitude``, ``--no-pose``, ``--no-median-shape``,
``--augment`` and ``--eval-frames``.

Usage
-----
    python models/landmarks/dcs/trainer.py --tag c1_base --iterations 8000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import LandmarkAugment                                     # noqa: E402
from backend import fit_backend                                          # noqa: E402
from data import LandmarkCache, TrialSampler, train_val_indices         # noqa: E402
from evaluate import embed_split, pair_metrics, source_probe            # noqa: E402
from geometry import GeometricFrontEnd                                   # noqa: E402
from losses import (SourceGRL, SubCenterArcMargin, adv_schedule,        # noqa: E402
                    contrastive, margin_schedule)
from model import DCSNet                                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Settings:
    tag: str = "c_base"
    iterations: int = 8000
    drivers: int = 32
    clips_per_driver: int = 8
    sources: int = 8
    # Plan §4 asks for crops in [96, 200], but the clips in this dataset are 100
    # frames (87,468 of 89,133 exactly 100). Anything above the clip length gets
    # reflect-padded, which makes the sequence a palindrome and forces the lagged
    # cross-correlations at +-L to mirror each other -- destroying the lead/lag
    # information Approach C is built on. So the range is capped at the clip length
    # and kept above the ~64 frames correlations need to be stable.
    crop_min: int = 80
    crop_max: int = 100
    lr: float = 5e-4
    lr_min: float = 1e-5
    weight_decay: float = 1e-3
    warmup: int = 500
    grad_clip: float = 5.0
    ema: float = 0.999
    seed: int = 0
    # model
    learned: int = 32
    kernel: int = 7
    embedding_dim: int = 256
    hidden: tuple[int, ...] = (512,)
    dropout: float = 0.3
    input_dropout: float = 0.2
    lags: tuple[int, ...] = (-8, -4, 4, 8)
    use_handcrafted: bool = True
    use_learned: bool = True
    use_lags: bool = True
    use_amplitude: bool = True
    # front-end
    pose: bool = True
    median_shape: bool = True
    derivatives: bool = True
    augment: bool = True
    # loss
    aam_margin: float = 0.2
    aam_scale: float = 30.0
    sub_centers: int = 3
    lambda_contrast: float = 0.5
    lambda_adv: float = 0.1
    shuffle_frac: float = 0.5
    # eval
    eval_every: int = 250
    eval_batch: int = 192
    backend: bool = True            # score with the §5 LDA+WCCN back-end as well
    backend_dims: int = 200
    backend_clips: int = 6000       # train clips used to fit the back-end
    mlflow: bool = True


def build(cfg: Settings, device, n_drivers: int, n_sources: int):
    fe = GeometricFrontEnd(derivatives=cfg.derivatives, median_shape=cfg.median_shape,
                           pose=cfg.pose).to(device)
    net = DCSNet(fe.out_channels, fe.handcrafted_channels(), learned=cfg.learned,
                 kernel=cfg.kernel, embedding_dim=cfg.embedding_dim, hidden=cfg.hidden,
                 dropout=cfg.dropout, input_dropout=cfg.input_dropout,
                 lags=cfg.lags, use_handcrafted=cfg.use_handcrafted,
                 use_learned=cfg.use_learned, use_lags=cfg.use_lags,
                 use_amplitude=cfg.use_amplitude).to(device)
    aam = SubCenterArcMargin(cfg.embedding_dim, n_drivers, cfg.sub_centers,
                             cfg.aam_scale, cfg.aam_margin).to(device)
    grl = SourceGRL(cfg.embedding_dim, n_sources).to(device)
    return fe, net, aam, grl


def lr_at(step: int, cfg: Settings) -> float:
    """Linear warm-up then cosine decay to ``lr_min``."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.iterations - cfg.warmup)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * min(1.0, p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c_base")
    ap.add_argument("--iterations", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, nargs="*", default=[512],
                    help="head widths; pass with no values for a direct linear head")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--input-dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--no-backend", action="store_true")
    ap.add_argument("--drivers", type=int, default=32)
    ap.add_argument("--clips-per-driver", type=int, default=8)
    ap.add_argument("--learned", type=int, default=32)
    ap.add_argument("--crop-min", type=int, default=80)
    ap.add_argument("--crop-max", type=int, default=100)
    ap.add_argument("--lambda-contrast", type=float, default=0.5)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--no-handcrafted", action="store_true")
    ap.add_argument("--no-learned", action="store_true")
    ap.add_argument("--no-lags", action="store_true")
    ap.add_argument("--no-amplitude", action="store_true")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--no-median-shape", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    cfg = Settings(
        tag=a.tag, iterations=a.iterations, seed=a.seed, lr=a.lr, drivers=a.drivers,
        clips_per_driver=a.clips_per_driver, learned=a.learned, crop_min=a.crop_min,
        crop_max=a.crop_max, lambda_contrast=a.lambda_contrast, lambda_adv=a.lambda_adv,
        eval_every=a.eval_every, hidden=tuple(a.hidden), dropout=a.dropout,
        input_dropout=a.input_dropout, weight_decay=a.weight_decay,
        backend=not a.no_backend, use_handcrafted=not a.no_handcrafted,
        use_learned=not a.no_learned, use_lags=not a.no_lags,
        use_amplitude=not a.no_amplitude, pose=not a.no_pose,
        median_shape=not a.no_median_shape, augment=not a.no_augment,
        mlflow=not a.no_mlflow)

    out = ROOT / "output/landmarks/dcs" / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = LandmarkCache("train")
    tr, va = train_val_indices(cache)
    sampler = TrialSampler(cache, tr, cfg.drivers, cfg.clips_per_driver, cfg.sources, cfg.seed)
    print(f"[{cfg.tag}] train {len(tr)} clips / {sampler.num_drivers} drivers / "
          f"{sampler.num_sources} sources, val {len(va)} clips", flush=True)

    fe, net, aam, grl = build(cfg, device, sampler.num_drivers, sampler.num_sources)
    aug = LandmarkAugment().to(device) if cfg.augment else None
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[{cfg.tag}] descriptor {net.descriptor_dim} dims, front-end {fe.out_channels} ch, "
          f"{n_par/1e6:.2f}M params", flush=True)

    decay, no_decay = [], []
    for m in (net, aam, grl):
        for name, p in m.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    run = None
    if cfg.mlflow:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=True)
            import mlflow
            mlflow.set_experiment("wba2026-landmarks-dcs")
            run = mlflow.start_run(run_name=cfg.tag)
            mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})
            mlflow.log_params({"descriptor_dim": net.descriptor_dim,
                               "front_end_channels": fe.out_channels, "params": n_par})
        except Exception as e:                                  # offline is not fatal
            print(f"[{cfg.tag}] mlflow disabled: {e}", flush=True)
            run = None

    # Fixed subset of train clips used to fit the §5 back-end at every validation,
    # so the back-end score is comparable across steps and across runs.
    backend_idx = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(1234))
                     [:cfg.backend_clips]]

    best = {"score3": -1.0, "selected": -1.0}
    t0 = time.time()
    for step in range(cfg.iterations):
        net.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        seqs, dlab, slab = sampler.batch()
        frames = random.randint(cfg.crop_min, cfg.crop_max)
        x = cache.gather(seqs, frames, device, random_crop=True)
        dlab, slab = dlab.to(device), slab.to(device)

        # The front-end and the augmentations have no learnable parameters, so they
        # run outside autograd: nothing upstream of the signal bank needs a graph,
        # and skipping it is most of the activation memory on a 4 GB GPU.
        with torch.no_grad():
            if aug is not None:
                x = aug(x)
            feats = fe(x)
            shuffled_feats = None
            if cfg.lambda_contrast > 0 and cfg.shuffle_frac > 0:
                # Time-shuffled copy of a subset: destroys temporal order while keeping
                # every per-frame statistic, so the push-apart term specifically forces
                # the embedding to depend on timing. Shuffling the *features* rather
                # than re-running the front-end on shuffled landmarks keeps the
                # per-frame marginals exactly identical and costs one permutation
                # instead of a second front-end pass.
                k = max(1, int(len(feats) * cfg.shuffle_frac))
                shuffled_feats = feats[:k][:, :, torch.randperm(feats.shape[2], device=device)]
        del x

        emb = net(feats)
        margin = margin_schedule(step, cfg.iterations, cfg.aam_margin)
        lam_adv = adv_schedule(step, cfg.iterations, cfg.lambda_adv)
        shuffled = net(shuffled_feats) if shuffled_feats is not None else None

        l_aam = aam(emb, dlab, margin)
        l_con = (contrastive(emb, dlab, slab, shuffled)
                 if cfg.lambda_contrast > 0 else emb.new_zeros(()))
        l_adv = grl(emb, slab, lam_adv) if cfg.lambda_adv > 0 else emb.new_zeros(())
        loss = l_aam + cfg.lambda_contrast * l_con + l_adv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], cfg.grad_clip)
        opt.step()

        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(cfg.ema).add_(v.float(), alpha=1 - cfg.ema)
                else:
                    ema[k] = v.clone().float()

        if step % 50 == 0:
            el = time.time() - t0
            print(f"[{cfg.tag}] iter {step+1}/{cfg.iterations} | loss {loss.item():.4f} "
                  f"| aam {l_aam.item():.4f} | con {l_con.detach().item():.4f} "
                  f"| grl {l_adv.detach().item():.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {el/(step+1):.2f} s/it", flush=True)
            if run:
                import mlflow
                mlflow.log_metrics({"loss": loss.item(), "loss_aam": l_aam.item(),
                                    "loss_contrast": l_con.detach().item(),
                                    "loss_grl": l_adv.detach().item(),
                                    "lr": opt.param_groups[0]["lr"], "margin": margin,
                                    "lambda_adv": lam_adv}, step=step)

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.iterations:
            # embed_split returns rows in length-bucket order, not in `va` order, so the
            # metadata must be taken in the order it reports back.
            val_emb, order = embed_split(net, fe, cache, va, device, cfg.eval_batch)
            val_rows = cache.meta.iloc[order.numpy()]
            m = pair_metrics(val_emb, val_rows)
            sp = source_probe(val_emb, val_rows)

            # §5 back-end, fitted on train embeddings only. Scored alongside plain
            # cosine rather than instead of it, because the submission could use
            # either and we need to know which one the weights actually favour.
            if cfg.backend:
                be_emb, be_order = embed_split(net, fe, cache, backend_idx, device,
                                               cfg.eval_batch)
                be = fit_backend(be_emb, cache.meta["driver_id"].to_numpy()[be_order.numpy()],
                                 dims=cfg.backend_dims)
                if be is not None:
                    mb = pair_metrics(be.apply(val_emb), val_rows)
                    m.update({"backend_score3": mb["score3"], "backend_score4": mb["score4"]})
            print(f"[{cfg.tag}] EVAL step {step+1} | 3-cond {m['score3']:.4f} | "
                  f"4-pair {m['score4']:.4f} | src-probe {sp:.3f}"
                  + (f" | backend 3-cond {m['backend_score3']:.4f} "
                     f"4-pair {m['backend_score4']:.4f}" if "backend_score3" in m else ""),
                  flush=True)
            if run:
                import mlflow
                mlflow.log_metrics({**{k: v for k, v in m.items() if not math.isnan(v)},
                                    "source_probe": sp}, step=step)
            # Select on whichever scoring we would actually submit: plain cosine or
            # the back-end. Tracking only the raw score would discard a checkpoint
            # that is better once LDA+WCCN is applied.
            selected = max(m["score3"], m.get("backend_score3", -1.0))
            if selected > best["selected"]:
                best = {**m, "step": step + 1, "source_probe": sp, "selected": selected,
                        "selected_by": "backend" if m.get("backend_score3", -1.0) > m["score3"]
                                       else "cosine"}
                torch.save({"model": net.state_dict(), "ema": ema, "config": asdict(cfg),
                            "metrics": best, "front_end": {"derivatives": cfg.derivatives,
                                                           "median_shape": cfg.median_shape,
                                                           "pose": cfg.pose}},
                           out / "best_model.pt")
                print(f"[{cfg.tag}] saved best selected {selected:.4f} "
                      f"({best['selected_by']})", flush=True)
            net.train()

    (out / "metrics.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
    print(f"[{cfg.tag}] DONE best selected {best['selected']:.4f} "
          f"({best.get('selected_by')}) | cosine 3-cond {best['score3']:.4f} 4-pair {best['score4']:.4f} "
          f"at step {best.get('step')} in {(time.time()-t0)/60:.1f} min", flush=True)
    if run:
        import mlflow
        mlflow.log_metrics({"best_score3": best["score3"], "best_score4": best["score4"],
                            "best_selected": best["selected"]})
        mlflow.log_artifact(str(out / "best_model.pt"))
        mlflow.end_run()


if __name__ == "__main__":
    main()
