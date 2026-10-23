"""§1.4 Generator-robustness augmentations, in landmark space, on GPU.

Each transform mimics something that actually differs between avatar generators,
which is the distribution shift the leaderboard different-generator score measures:

    expression gain   generators under- or over-transfer expression amplitude
    temporal low-pass generator-side temporal smoothing
    landmark jitter   detector noise on rendered frames, plus occasional spikes
    speed perturb     frame-rate / resampling differences
    frame drop/dup    encoding artefacts
    global similarity small roll and scale, applied before the front-end so the
                      pose channels stay realistic rather than being hand-edited

Randomness is drawn **per sample**, not per batch: one gain for 256 clips would
make the whole batch share a nuisance factor, which a contrastive loss can simply
subtract out instead of learning to ignore.

Deliberately disjoint from the robustness proxy used for evaluation, so the proxy
keeps measuring generalisation rather than memorised augmentation.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from geometry import LANDMARK_REGIONS, LEFT_EYE_IDX, RIGHT_EYE_IDX  # noqa: F401


class LandmarkAugment(nn.Module):
    def __init__(self, gain: tuple[float, float] = (0.7, 1.2), smooth_sigma: float = 1.5,
                 jitter: tuple[float, float] = (0.005, 0.02), speed: tuple[float, float] = (0.9, 1.1),
                 drop_rate: float = 0.05, roll_deg: float = 10.0,
                 scale: tuple[float, float] = (0.9, 1.1), spike_rate: float = 0.01,
                 smooth_levels: int = 3,
                 enabled: tuple[str, ...] = ("gain", "smooth", "jitter", "speed", "drop", "similarity")) -> None:
        super().__init__()
        self.gain, self.smooth_sigma, self.jitter = gain, smooth_sigma, jitter
        self.speed, self.drop_rate = speed, drop_rate
        self.roll_deg, self.scale, self.spike_rate = roll_deg, scale, spike_rate
        self.smooth_levels = smooth_levels
        self.enabled = set(enabled)
        # Region id per landmark, so the per-region gain is one gather instead of a
        # Python loop over regions.
        region_of = torch.zeros(106, dtype=torch.long)
        for r, (name, idx) in enumerate(LANDMARK_REGIONS.items()):
            region_of[torch.tensor(idx, dtype=torch.long)] = r
        self.register_buffer("_region_of", region_of)
        self.n_regions = len(LANDMARK_REGIONS)

    @staticmethod
    def _u(b: int, lo: float, hi: float, device) -> Tensor:
        return torch.rand(b, device=device) * (hi - lo) + lo

    def _similarity(self, x: Tensor) -> Tensor:
        """Per-sample roll and scale about the clip centroid."""
        b = x.shape[0]
        ang = self._u(b, -math.radians(self.roll_deg), math.radians(self.roll_deg), x.device)
        s = self._u(b, *self.scale, x.device)
        c, sn = torch.cos(ang) * s, torch.sin(ang) * s
        rot = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)  # [B,2,2]
        cen = x.mean(dim=(1, 2), keepdim=True)
        return torch.einsum("btvi,bji->btvj", x - cen, rot) + cen

    def _gain(self, x: Tensor) -> Tensor:
        """Per-sample, per-region gain on the deformation about the clip-median shape."""
        med = x.median(dim=1, keepdim=True).values
        g = self._u(x.shape[0] * self.n_regions, *self.gain, x.device).view(-1, self.n_regions)
        g = g[:, self._region_of]                                          # [B,106]
        return med + (x - med) * g[:, None, :, None]

    def _smooth(self, x: Tensor) -> Tensor:
        """Per-sample Gaussian temporal low-pass.

        Sigma is drawn per sample then snapped to one of ``smooth_levels`` values, so
        the whole batch needs a handful of cheap depthwise convs plus one gather,
        rather than a grouped conv with B*V*C separate kernels.
        """
        b, t, v, c = x.shape
        sigma = self._u(b, 0.0, self.smooth_sigma, x.device)
        levels = torch.linspace(0.0, self.smooth_sigma, self.smooth_levels, device=x.device)
        pick = (sigma / self.smooth_sigma * (self.smooth_levels - 1)).round().long()

        half = max(1, int(4 * self.smooth_sigma + 1) // 2)
        taps = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
        flat = x.permute(0, 2, 3, 1).reshape(b, v * c, t)
        padded = F.pad(flat, (half, half), mode="replicate")

        outs = [flat]                                                      # level 0 = identity
        for lv in levels[1:].tolist():
            k = torch.exp(-0.5 * (taps / lv).square())
            k = (k / k.sum()).view(1, 1, -1).expand(v * c, 1, taps.numel())
            outs.append(F.conv1d(padded, k, groups=v * c))
        stacked = torch.stack(outs, dim=0)                                 # [L,B,V*C,T]
        sel = stacked[pick, torch.arange(b, device=x.device)]               # [B,V*C,T]
        return sel.view(b, v, c, t).permute(0, 3, 1, 2)

    def _jitter(self, x: Tensor) -> Tensor:
        """IOD-proportional Gaussian noise, plus rare single-frame spikes."""
        left = x[:, :, LEFT_EYE_IDX, :].mean(dim=2)
        right = x[:, :, RIGHT_EYE_IDX, :].mean(dim=2)
        iod = (left - right).norm(dim=-1).mean(dim=1).clamp_min(1e-6)       # [B]
        frac = self._u(x.shape[0], *self.jitter, x.device)
        scale = (iod * frac).view(-1, 1, 1, 1)
        if self.spike_rate > 0:
            # One noise draw serves both terms; a second randn_like over the whole
            # batch costs more than the independence buys.
            hit = (torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device) < self.spike_rate)
            scale = scale + hit * (iod * 0.05).view(-1, 1, 1, 1)
        return x + torch.randn_like(x) * scale

    def _speed(self, x: Tensor) -> Tensor:
        """Per-sample linear resampling, keeping the output length fixed at T.

        Length is preserved so the batch stays one tensor; a slower speed therefore
        reads a shorter span of the crop and a faster one reflects at the end.
        """
        b, t = x.shape[0], x.shape[1]
        if t < 2:
            return x
        sp = self._u(b, *self.speed, x.device)
        pos = torch.arange(t, device=x.device).float()[None, :] * sp[:, None]
        period = max(1.0, 2.0 * t - 2.0)
        pos = pos % period
        pos = torch.where(pos >= t, period - pos, pos).clamp(0, t - 1)
        i0 = pos.floor().long().clamp(max=t - 2)
        frac = (pos - i0.float())[..., None, None]
        bi = torch.arange(b, device=x.device)[:, None]
        return x[bi, i0] * (1 - frac) + x[bi, i0 + 1] * frac

    def _drop(self, x: Tensor) -> Tensor:
        """Per-sample frame drop/duplicate: replace a frame by its neighbour."""
        b, t = x.shape[0], x.shape[1]
        if t < 4 or self.drop_rate <= 0:
            return x
        idx = torch.arange(t, device=x.device)[None, :].expand(b, t).clone()
        hit = torch.rand(b, t, device=x.device) < self.drop_rate
        idx = torch.where(hit, (idx - 1).clamp_min(0), idx)
        return x[torch.arange(b, device=x.device)[:, None], idx]

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """[B,T,106,2] -> [B,T,106,2]. Order matters: geometry, then amplitude, then sampling."""
        if "similarity" in self.enabled:
            x = self._similarity(x)
        if "gain" in self.enabled:
            x = self._gain(x)
        if "smooth" in self.enabled:
            x = self._smooth(x)
        if "jitter" in self.enabled:
            x = self._jitter(x)
        if "speed" in self.enabled:
            x = self._speed(x)
        if "drop" in self.enabled:
            x = self._drop(x)
        return x
