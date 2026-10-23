"""Loss stack shared with plan §2:  L = L_AAM + lambda_c * L_contrast + lambda_adv * L_srcGRL

The three terms attack three different failure modes:

*  **Sub-centre AAM-softmax** over train driver IDs gives a large-margin embedding.
   Three sub-centres per driver because a driver genuinely looks different per mode
   and per target, and forcing one centroid blurs all of them together.
*  **Contrastive**, three terms: pull clips of the same driver together; push apart
   clips that share a *target* but have different drivers (the hard negatives the
   test trials are built from); and push a clip away from a time-shuffled copy of
   itself, which forces the embedding to depend on temporal order rather than on
   order-free statistics.
*  **Source GRL** reverses gradients into a classifier that tries to predict the
   target/source identity from the embedding, so the encoder is penalised for
   leaking the avatar appearance — the documented shortcut in this task.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function


class SubCenterArcMargin(nn.Module):
    """Sub-centre ArcFace head. Margin is warmed up linearly by the caller."""

    def __init__(self, dim: int, classes: int, sub_centers: int = 3, scale: float = 30.0,
                 margin: float = 0.2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes * sub_centers, dim))
        nn.init.xavier_uniform_(self.weight)
        self.classes, self.k = classes, sub_centers
        self.scale, self.margin = scale, margin

    def forward(self, emb: Tensor, target: Tensor, margin: float | None = None) -> Tensor:
        m = self.margin if margin is None else margin
        cos = F.linear(F.normalize(emb), F.normalize(self.weight))
        cos = cos.view(-1, self.classes, self.k).max(dim=2).values     # nearest sub-centre
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = cos.acos()
        # Only the target logit gets the additive angular margin.
        hot = F.one_hot(target, self.classes).bool()
        adjusted = torch.where(hot, (theta + m).clamp(max=math.pi).cos(), cos)
        return F.cross_entropy(adjusted * self.scale, target)


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return -ctx.lambd * grad, None


class SourceGRL(nn.Module):
    """Predicts the target/source id through a gradient-reversal layer."""

    def __init__(self, dim: int, sources: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, sources))

    def forward(self, emb: Tensor, source: Tensor, lambd: float) -> Tensor:
        return F.cross_entropy(self.net(_GradReverse.apply(emb, lambd)), source)


def contrastive(emb: Tensor, driver: Tensor, source: Tensor,
                shuffled: Tensor | None = None, margin: float = 0.3) -> Tensor:
    """Three-term contrastive loss on L2-normalised embeddings.

    Term 1 pulls same-driver pairs together. Term 2 pushes same-source /
    different-driver pairs apart with a margin — these mirror the impostor trials,
    which always share the target appearance. Term 3 pushes each clip away from a
    time-shuffled version of itself; ``shuffled`` may cover only the first rows of
    the batch, since shuffling the whole batch doubles the forward cost for a term
    that converges quickly.
    """
    e = F.normalize(emb, p=2, dim=-1)
    sim = e @ e.T
    same_d = driver[:, None] == driver[None, :]
    same_s = source[:, None] == source[None, :]
    eye = torch.eye(len(e), dtype=torch.bool, device=e.device)

    pos = same_d & ~eye
    neg = same_s & ~same_d
    loss = e.new_zeros(())
    if bool(pos.any()):
        loss = loss + (1.0 - sim[pos]).mean()
    if bool(neg.any()):
        loss = loss + (sim[neg] + margin).clamp_min(0).mean()
    if shuffled is not None and len(shuffled):
        s = F.normalize(shuffled, p=2, dim=-1)
        k = len(s)
        loss = loss + ((e[:k] * s).sum(-1) + margin).clamp_min(0).mean()
    return loss


def margin_schedule(step: int, total: int, peak: float, warmup_frac: float = 0.2) -> float:
    """Linear warm-up of the AAM margin over the first ``warmup_frac`` of training."""
    w = max(1, int(total * warmup_frac))
    return peak * min(1.0, step / w)


def adv_schedule(step: int, total: int, peak: float, ramp_frac: float = 0.3) -> float:
    """Ramp the GRL weight 0 -> peak, so the encoder learns something before being attacked."""
    r = max(1, int(total * ramp_frac))
    return peak * min(1.0, step / r)
