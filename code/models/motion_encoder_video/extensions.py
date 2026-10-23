"""Small video-only experimental components, initialized from scratch."""
import torch
from torch import nn
from torch.nn import functional as F


def cross_mode_positive_loss(z, labels, modes, temperature):
    """Same-driver, opposite-mode positives; all other clips in denominator.

    Exclude anchors without a cross-mode positive. Return None if none qualify.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if labels.shape != modes.shape or labels.shape != (len(z),):
        raise ValueError("labels and modes must contain one value per embedding")
    if not torch.all((modes == 0) | (modes == 1)):
        raise ValueError("modes must be 0 (speaking) or 1 (listening)")
    z = F.normalize(z.float(), dim=-1)
    positives = labels[:, None].eq(labels[None, :]) & modes[:, None].ne(modes[None, :])
    counts = positives.sum(1)
    valid = counts > 0
    if not valid.any():
        return None
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(len(z), device=z.device, dtype=torch.bool), -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0).sum(1)[valid] / counts[valid]).mean()


class RegionalTemporalBranch(nn.Module):
    """Four spatial regions, 43-step temporal receptive field, mean/std pooling.

    Input is [B,C,H,W,T]. No test-batch statistics or pair inputs.
    """
    def __init__(self, channels, output_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, 3, padding=d, dilation=d, groups=channels),
                          nn.Conv1d(channels, channels, 1), nn.GroupNorm(1, channels), nn.ReLU())
            for d in (1, 4, 16)
        ])
        self.project = nn.Linear(channels * 4 * 2, output_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, h):
        b, c, _, _, t = h.shape
        x = F.adaptive_avg_pool3d(h, (2, 2, t)).reshape(b, c, 4, t)
        x = x.permute(0, 2, 1, 3).reshape(b * 4, c, t)
        for block in self.blocks:
            x = x + block(x)
        stats = torch.cat((x.mean(-1), (x.float().var(-1, unbiased=False) + 1e-5).sqrt().to(x.dtype)), dim=1)
        return self.gate * self.project(stats.reshape(b, -1))
