"""§1.6 Fast evaluation — one code path shared by validation and inference.

No sliding windows: the correlation signature pools over whole sequences, so each
clip is embedded in a single forward. Sequences are grouped into buckets of
*identical* length (98% of clips are exactly 100 frames), which keeps every
clip-level statistic exact without mask bookkeeping and still gets the whole
validation split done in a handful of forwards.

``postprocess`` is the single place embedding-level transforms live, so validation
and ``infer.py`` cannot drift apart — plan §1.6 note 4 and §5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MODES = ("speaking", "listening")


def compute_auc(scores: Tensor, targets: Tensor) -> float:
    """Trapezoidal ROC AUC. Identical to the repo-wide implementation."""
    pos, neg = int(targets.sum()), int((1 - targets).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    t = targets[scores.argsort(descending=True)]
    tpr, fpr = t.cumsum(0).float() / pos, (1 - t).cumsum(0).float() / neg
    zero = torch.zeros(1, device=tpr.device)
    return torch.trapz(torch.cat((zero, tpr)), torch.cat((zero, fpr))).item()


def compute_eer(scores: Tensor, targets: Tensor) -> float:
    pos, neg = int(targets.sum()), int((1 - targets).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    t = targets[scores.argsort(descending=True)]
    tpr, fpr = t.cumsum(0).float() / pos, (1 - t).cumsum(0).float() / neg
    diff = fpr - (1 - tpr)
    idx = (diff[:-1] * diff[1:] <= 0).nonzero(as_tuple=True)[0]
    return float(fpr[idx[0]]) if len(idx) else float("nan")


def official_score(m: dict[str, float]) -> float:
    """Repo convention: mean of speaking, listening and cross (mean of the two cross pairs)."""
    g = lambda k: m.get(f"emb_{k}_auc", float("nan"))  # noqa: E731
    cross = (g("speaking_vs_listening") + g("listening_vs_speaking")) / 2
    vals = [v for v in (g("speaking_vs_speaking"), g("listening_vs_listening"), cross)
            if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else 0.0


def mean4(m: dict[str, float]) -> float:
    """Plain mean of the 4 probe/gallery mode pairs — this is what the leaderboard averages."""
    vals = [m.get(f"emb_{a}_vs_{b}_auc", float("nan")) for a in MODES for b in MODES]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else 0.0


@dataclass
class PostStats:
    """Embedding-level transform fitted on train embeddings only (§5)."""
    mean: Tensor | None = None


def fit_postprocess(train_emb: Tensor) -> PostStats:
    return PostStats(mean=train_emb.mean(dim=0, keepdim=True))


def postprocess(emb: Tensor, stats: PostStats | None = None) -> Tensor:
    """Centre (if fitted) then L2-normalise. Called by both evaluate() and infer.py.

    Only train-fitted statistics are used: no test-set mean, no per-generation
    normalisation, nothing that would depend on seeing the evaluation set.
    """
    if stats is not None and stats.mean is not None:
        emb = emb - stats.mean.to(emb.device, emb.dtype)
    return F.normalize(emb, p=2, dim=-1)


@torch.no_grad()
def embed_split(model: nn.Module, front_end: nn.Module, cache, seq_idx: Tensor,
                device: torch.device, max_batch: int = 256,
                autocast: bool = False) -> tuple[Tensor, Tensor]:
    """Embed every sequence in ``seq_idx``, whole-sequence, bucketed by exact length.

    Returns (embeddings, seq_idx_in_embedding_order).
    """
    model.eval()
    embs, order = [], []
    for bucket in cache.length_buckets(seq_idx, max_batch):
        x = cache.gather_full(bucket, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
            e = model(front_end(x))
        embs.append(e.float().cpu())
        order.append(bucket)
    return torch.cat(embs), torch.cat(order)


def pair_metrics(emb: Tensor, meta_rows, stats: PostStats | None = None) -> dict[str, float]:
    """AUC and EER for each of the 4 probe/gallery mode pairs.

    ``meta_rows`` is the metadata slice aligned with ``emb`` (same order), carrying
    ``set``, ``mode`` and ``driver_id``.
    """
    e = postprocess(emb, stats)
    sim = e @ e.T
    drivers = meta_rows["driver_id"].to_numpy()
    sets = meta_rows["set"].to_numpy()
    modes = meta_rows["mode"].to_numpy()
    uniq = {d: i for i, d in enumerate(sorted(set(drivers)))}
    lab = torch.tensor([uniq[d] for d in drivers])

    idx = {(s, m): torch.tensor([i for i in range(len(e)) if sets[i] == s and modes[i] == m])
           for s in ("probe", "gallery") for m in MODES}
    out: dict[str, float] = {}
    for pm in MODES:
        for gm in MODES:
            p, g = idx[("probe", pm)], idx[("gallery", gm)]
            if len(p) == 0 or len(g) == 0:
                continue
            scores = sim[p][:, g].flatten()
            targets = (lab[p][:, None] == lab[g][None, :]).flatten().float()
            out[f"emb_{pm}_vs_{gm}_auc"] = compute_auc(scores, targets)
            out[f"emb_{pm}_vs_{gm}_eer"] = compute_eer(scores, targets)
    out["score3"] = official_score(out)
    out["score4"] = mean4(out)
    return out


@torch.no_grad()
def source_probe(emb: Tensor, meta_rows) -> float:
    """How well the target/source id can be read back off the embedding.

    A nearest-centroid probe, not a trained classifier: cheap enough to log every
    validation step. High accuracy means the appearance shortcut is still present,
    which is the thing the GRL term is supposed to suppress.
    """
    e = F.normalize(emb, p=2, dim=-1)
    src = meta_rows["source_id"].to_numpy()
    names = sorted(set(src))
    if len(names) < 2:
        return float("nan")
    cents = torch.stack([e[[i for i in range(len(e)) if src[i] == n]].mean(0) for n in names])
    pred = (e @ F.normalize(cents, p=2, dim=-1).T).argmax(dim=1)
    truth = torch.tensor([names.index(s) for s in src])
    return float((pred == truth).float().mean())
