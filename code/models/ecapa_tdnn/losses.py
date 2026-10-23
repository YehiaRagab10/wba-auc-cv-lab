"""Loss functions for Kinematic-TDNN training.

Contains:
    - SubCenterArcMarginProduct: Sub-center AAM-softmax head
    - ContrastiveLoss: 3-term contrastive (pull / push / time-shuffle)
    - GradientReversalLayer + SourceClassifier: domain-adversarial source GRL
    - KTDNNLoss: combined L = L_AAM + λ_c·L_contrast + λ_adv·L_srcGRL
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Pairwise distance helper
# ---------------------------------------------------------------------------

def pairwise_squared_distance(x: Tensor, y: Tensor | None = None) -> Tensor:
    """Efficient pairwise squared Euclidean distance."""
    y = x if y is None else y
    return (
        x.square().sum(-1, keepdim=True)
        + y.square().sum(-1).unsqueeze(-2)
        - 2 * x @ y.transpose(-2, -1)
    ).clamp_min_(0)


# ---------------------------------------------------------------------------
# Sub-center AAM-Softmax
# ---------------------------------------------------------------------------

class SubCenterArcMarginProduct(nn.Module):
    """Sub-center Additive Angular Margin Softmax (AAM-softmax).

    Args:
        in_features: embedding dimension
        out_features: number of classes (train driver IDs)
        s: scale factor
        m: angular margin (final value; warmed up from 0)
        k: number of sub-centers per class
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.2,
        k: int = 3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.k = k
        # Weight: [out_features * k, in_features]
        self.weight = nn.Parameter(torch.empty(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, embeddings: Tensor, labels: Tensor, margin_scale: float = 1.0
    ) -> Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalised embeddings
            labels: [B] integer class labels
            margin_scale: 0..1, linear warm-up multiplier for the margin

        Returns:
            logits: [B, num_classes] scaled cosine similarities with margin
        """
        # Normalise weights
        W = F.normalize(self.weight, p=2, dim=1)  # [out*k, D]

        # Cosine similarity
        cosine = F.linear(embeddings, W)  # [B, out*k]

        # Reshape to [B, out, k] and take max over sub-centers
        cosine = cosine.view(-1, self.out_features, self.k)
        cosine, _ = cosine.max(dim=2)  # [B, out]

        # Apply margin to the target class
        m = self.m * margin_scale
        if m > 0:
            # Numerically stable arc margin
            cosine_of_target = cosine.gather(1, labels.unsqueeze(1))  # [B, 1]
            theta = cosine_of_target.acos().clamp(0, math.pi)
            marginal_target = (theta + m).cos()
            cosine = cosine.scatter(1, labels.unsqueeze(1), marginal_target)

        return cosine * self.s


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 0.0

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return _GradReverse.apply(x, self.alpha)


class SourceClassifier(nn.Module):
    """2-layer MLP predicting source (target) ID through a GRL.

    Kept outside the embedding model's state_dict so inference doesn't
    need it.
    """

    def __init__(self, embedding_dim: int, num_sources: int, hidden: int = 512) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_sources),
        )

    def set_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: [B, D] (L2-normalised or raw — GRL handles either)

        Returns:
            logits: [B, num_sources]
        """
        return self.mlp(self.grl(embeddings))


# ---------------------------------------------------------------------------
# 3-term contrastive loss (NVIDIA-style)
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """3-term contrastive loss: pull same-driver, push same-source/diff-driver,
    push time-shuffled same-driver.

    Operates on L2-normalised clip-level embeddings. Computed in log-space
    via logsumexp for numerical stability.

    Args:
        temperature: temperature scaling for similarity
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: Tensor,
        shuffled: Tensor | None,
        drivers: Tensor,
        sources: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            embeddings: [B, D] L2-normalised
            shuffled: [B, D] L2-normalised embeddings from time-shuffled input, or None
            drivers: [B] driver IDs
            sources: [B] source IDs

        Returns:
            loss, info_dict
        """
        B = embeddings.shape[0]
        dist = pairwise_squared_distance(embeddings.float())  # [B, B]
        scale = dist.mean().clamp_min(1e-3)
        log_sim = -dist / (scale * self.temperature)

        # Mask out self-pairs
        log_sim.fill_diagonal_(-torch.inf)

        same_driver = drivers.unsqueeze(1) == drivers.unsqueeze(0)  # [B, B]
        same_source = sources.unsqueeze(1) == sources.unsqueeze(0)  # [B, B]

        # Pull: same driver (excluding self)
        pull_mask = same_driver.clone()
        pull_mask.fill_diagonal_(False)
        log_pull = torch.logsumexp(
            log_sim.masked_fill(~pull_mask, -torch.inf), dim=-1
        )  # [B]

        # Push: same source, different driver
        push_mask = same_source & ~same_driver
        log_push = torch.logsumexp(
            log_sim.masked_fill(~push_mask, -torch.inf), dim=-1
        )  # [B]

        logits = [log_pull, log_push]

        # Time-shuffle push: same driver, shuffled copy
        if shuffled is not None:
            dist_sh = pairwise_squared_distance(
                embeddings.float(), shuffled.float()
            )  # [B, B]
            log_sim_sh = -dist_sh / (scale * self.temperature)
            sh_mask = same_driver
            log_shuffle = torch.logsumexp(
                log_sim_sh.masked_fill(~sh_mask, -torch.inf), dim=-1
            )
            logits.append(log_shuffle)

        log_prob = F.log_softmax(torch.stack(logits, dim=0), dim=0)[0]  # [B]

        finite = torch.isfinite(log_prob)
        if finite.all():
            loss = -log_prob.mean()
        elif finite.any():
            loss = -log_prob[finite].mean()
        else:
            loss = torch.zeros((), device=embeddings.device, requires_grad=True)

        info = {
            "mean_dist": dist.mean().detach(),
            "pull": log_pull.exp().mean().detach(),
            "push": log_push.exp().mean().detach(),
            "p": log_prob.exp().mean().detach(),
        }
        return loss, info


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class KTDNNLoss(nn.Module):
    """Combined loss: L = L_AAM + λ_c·L_contrast + λ_adv·L_srcGRL.

    The AAM head and source classifier are owned by this module and saved
    separately from the embedding model.

    Args:
        embedding_dim: dimension of the embedding
        num_drivers: number of unique driver IDs (for AAM)
        num_sources: number of unique source IDs (for GRL)
        s, m, k: AAM-softmax hyperparams
        temperature: contrastive loss temperature
        lambda_contrast: weight for contrastive loss
        lambda_adv_max: maximum weight for GRL loss (ramped from 0)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_drivers: int,
        num_sources: int,
        s: float = 30.0,
        m: float = 0.2,
        k: int = 3,
        temperature: float = 0.5,
        lambda_contrast: float = 0.5,
        lambda_adv_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.aam_head = SubCenterArcMarginProduct(
            embedding_dim, num_drivers, s=s, m=m, k=k
        )
        self.contrastive = ContrastiveLoss(temperature)
        self.source_classifier = SourceClassifier(
            embedding_dim, num_sources
        )
        self.lambda_contrast = lambda_contrast
        self.lambda_adv_max = lambda_adv_max

    def forward(
        self,
        embeddings: Tensor,
        shuffled_embeddings: Tensor | None,
        driver_labels: Tensor,
        source_labels: Tensor,
        margin_scale: float = 1.0,
        adv_scale: float = 0.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Args:
            embeddings: [B, D] raw embeddings (pre-L2-norm, from BN layer)
            shuffled_embeddings: [B, D] from time-shuffled input, or None
            driver_labels: [B] contiguous driver class indices
            source_labels: [B] contiguous source class indices
            margin_scale: 0..1 for AAM margin warm-up
            adv_scale: 0..1 for GRL lambda ramp

        Returns:
            total_loss, info_dict
        """
        info: dict[str, float] = {}

        # L2-normalise for contrastive and AAM
        emb_norm = F.normalize(embeddings, p=2, dim=-1)

        # --- AAM-softmax (original clips only) ---
        aam_logits = self.aam_head(emb_norm, driver_labels, margin_scale)
        loss_aam = F.cross_entropy(aam_logits, driver_labels)
        info["loss_aam"] = loss_aam.item()

        # --- Contrastive loss ---
        shuf_norm = (
            F.normalize(shuffled_embeddings, p=2, dim=-1)
            if shuffled_embeddings is not None
            else None
        )
        loss_contrast, contrast_info = self.contrastive(
            emb_norm, shuf_norm, driver_labels, source_labels
        )
        info["loss_contrast"] = loss_contrast.item()
        info.update({f"contrast_{k}": v.item() if isinstance(v, Tensor) else v
                     for k, v in contrast_info.items()})

        # --- Source GRL ---
        # GRL alpha carries the adversarial scale; classifier loss is weighted at 1.0
        # so the classifier trains at full strength while the encoder gradient is reversed
        # and scaled by lambda_adv.
        lambda_adv = self.lambda_adv_max * adv_scale
        self.source_classifier.set_alpha(lambda_adv)
        src_logits = self.source_classifier(emb_norm)
        loss_src = F.cross_entropy(src_logits, source_labels)
        info["loss_src_grl"] = loss_src.item()

        # --- Total ---
        total = loss_aam + self.lambda_contrast * loss_contrast + loss_src
        info["loss_total"] = total.item()

        return total, info