"""§4 Approach C — Dynamic Correlation Signature (DCS) network.

The premise: a driver identity lives in *how* motion channels co-vary — smile with
head tilt, blink with brow, jaw with lip corners, and the relative timing between
them — not in absolute amplitude. Pearson correlations and normalised covariances
are invariant to per-channel gain and offset, which is precisely the distortion a
different avatar generator introduces. That makes this the one descriptor here
whose robustness to the leaderboard different-generator split is structural rather
than hoped for.

Descriptor per sequence (3784 dims, no PCA anywhere):

    1128  correlation upper triangle of the 48 handcrafted signals (6 pose + 42 region)
     496  correlation upper triangle of the 32 learned signals
    1984  lagged cross-correlation of the learned signals at lags {-8,-4,+4,+8}
      80  log-std of every signal          (amplitude, kept separable)
      96  band energy of the learned signals in 3 bands (amplitude)

All of it is differentiable, so the learned signal bank — a single Conv1d that
discovers its own action units from the 843-channel front-end — is trained through
the correlation ops end to end.

Amplitude features are grouped behind one flag because the plan predicts they help
within a generator and hurt across generators; that is an ablation we want to be
able to run by itself.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FPS = 30.0
BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0))   # head sway / speech rhythm / blinks


def _upper_tri(m: Tensor) -> Tensor:
    """Strict upper triangle of a batch of square matrices. [B,M,M] -> [B,M(M-1)/2]"""
    i, j = torch.triu_indices(m.shape[-1], m.shape[-1], offset=1, device=m.device)
    return m[:, i, j]


def _fisher_z(c: Tensor) -> Tensor:
    """Fisher z-transform. Correlations near +-1 are clamped so atanh stays finite."""
    return torch.atanh(c.clamp(-0.999, 0.999))


class CorrelationSignature(nn.Module):
    """Turns ``[B, M, T]`` signals into correlation / lag / amplitude statistics."""

    def __init__(self, lags: tuple[int, ...] = (-8, -4, 4, 8)) -> None:
        super().__init__()
        self.lags = lags

    @staticmethod
    def _centred(s: Tensor) -> tuple[Tensor, Tensor]:
        """Mean-remove along time and return the per-signal standard deviation."""
        x = s - s.mean(dim=-1, keepdim=True)
        sd = x.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        return x, sd

    def correlation(self, s: Tensor) -> Tensor:
        """Fisher-z upper triangle of the Pearson correlation matrix. -> [B, M(M-1)/2]"""
        x, sd = self._centred(s)
        c = (x @ x.transpose(1, 2)) / x.shape[-1]
        c = c / (sd[:, :, None] * sd[:, None, :])
        return _fisher_z(_upper_tri(c))

    def lagged(self, s: Tensor) -> Tensor:
        """Fisher-z upper triangle of the lagged cross-correlation, one block per lag.

        At lag L, entry (i, j) correlates signal i at t with signal j at t+L, so the
        upper triangle at +L and at -L together carry both lead and lag directions
        (the matrix is not symmetric). This is what encodes, e.g., brows leading mouth.
        """
        x, sd = self._centred(s)
        t = x.shape[-1]
        out = []
        for lag in self.lags:
            k = abs(lag)
            if t <= k + 1:                       # sequence too short for this lag
                out.append(torch.zeros(x.shape[0], x.shape[1] * (x.shape[1] - 1) // 2,
                                       device=x.device, dtype=x.dtype))
                continue
            a, b = (x[:, :, :t - k], x[:, :, k:]) if lag > 0 else (x[:, :, k:], x[:, :, :t - k])
            c = (a @ b.transpose(1, 2)) / (t - k)
            c = c / (sd[:, :, None] * sd[:, None, :])
            out.append(_fisher_z(_upper_tri(c)))
        return torch.cat(out, dim=-1)

    @staticmethod
    def log_std(s: Tensor) -> Tensor:
        """Per-signal log standard deviation — the amplitude the correlations discard."""
        return s.std(dim=-1).clamp_min(1e-6).log()

    @staticmethod
    def band_energy(s: Tensor) -> Tensor:
        """Log energy in 3 temporal bands via rFFT. -> [B, M*3]

        Band edges are recomputed from the actual sequence length, so a whole-sequence
        forward at test time and a short training crop describe the same frequencies.
        """
        t = s.shape[-1]
        x = s - s.mean(dim=-1, keepdim=True)
        power = torch.fft.rfft(x, dim=-1).abs().square()           # [B,M,t//2+1]
        freq = torch.arange(power.shape[-1], device=s.device) * (FPS / max(t, 1))
        out = []
        for lo, hi in BANDS:
            sel = (freq >= lo) & (freq < hi)
            if not bool(sel.any()):
                sel = freq == freq.min()
            out.append(power[:, :, sel].sum(dim=-1).clamp_min(1e-8).log())
        return torch.cat(out, dim=-1)


class DCSNet(nn.Module):
    """Front-end features -> 256-d embedding through a correlation signature.

    ``front_end`` is applied by the caller (it is shared with the other approaches),
    so this module takes ``[B, C, T]`` features and the indices of the handcrafted
    signal channels within them.
    """

    def __init__(self, in_channels: int, handcrafted_idx: Tensor, learned: int = 32,
                 kernel: int = 7, embedding_dim: int = 256, hidden: tuple[int, ...] = (1024, 512),
                 dropout: float = 0.2, lags: tuple[int, ...] = (-8, -4, 4, 8),
                 use_handcrafted: bool = True, use_learned: bool = True,
                 use_lags: bool = True, use_amplitude: bool = True,
                 input_dropout: float = 0.0) -> None:
        super().__init__()
        assert use_handcrafted or use_learned, "DCS needs at least one signal set"
        self.register_buffer("handcrafted_idx", handcrafted_idx.clone())
        self.use_handcrafted, self.use_learned = use_handcrafted, use_learned
        self.use_lags, self.use_amplitude = use_lags, use_amplitude
        self.sig = CorrelationSignature(lags)

        # Input BatchNorm over the 843 front-end channels: their scales differ by
        # orders of magnitude (pixel-unit deformation vs. log-area vs. radians).
        self.input_bn = nn.BatchNorm1d(in_channels)
        if use_learned:
            self.bank = nn.Sequential(
                nn.Conv1d(in_channels, learned, kernel, padding=kernel // 2),
                nn.BatchNorm1d(learned),
                nn.Tanh(),            # bounded, so correlations stay well conditioned
            )

        m_h, m_l = handcrafted_idx.numel(), learned
        dim = 0
        if use_handcrafted:
            dim += m_h * (m_h - 1) // 2
            if use_amplitude:
                dim += m_h
        if use_learned:
            dim += m_l * (m_l - 1) // 2
            if use_lags:
                dim += len(lags) * m_l * (m_l - 1) // 2
            if use_amplitude:
                dim += m_l + m_l * len(BANDS)
        self.descriptor_dim = dim

        # Head width is configurable because the plan default (1024, 512) is 4.7M
        # parameters over 181 training drivers, which memorises them: measured
        # held-in 0.8542 vs held-out 0.5906. Dropout directly on the descriptor acts
        # like feature bagging over the 1128 handcrafted correlations, which is a
        # better-matched regulariser than dropout after the first projection.
        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        prev = dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, embedding_dim), nn.BatchNorm1d(embedding_dim)]
        self.head = nn.Sequential(*layers)

    def descriptor(self, features: Tensor) -> Tensor:
        """[B, C, T] front-end features -> [B, descriptor_dim]."""
        f = self.input_bn(features)
        parts = []
        if self.use_handcrafted:
            s_h = f[:, self.handcrafted_idx]
            parts.append(self.sig.correlation(s_h))
            if self.use_amplitude:
                parts.append(self.sig.log_std(s_h))
        if self.use_learned:
            s_l = self.bank(f)
            parts.append(self.sig.correlation(s_l))
            if self.use_lags:
                parts.append(self.sig.lagged(s_l))
            if self.use_amplitude:
                parts += [self.sig.log_std(s_l), self.sig.band_energy(s_l)]
        d = torch.cat(parts, dim=-1)
        return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, features: Tensor) -> Tensor:
        return self.head(self.descriptor(features))


def embed(model: DCSNet, front_end: nn.Module, landmarks: Tensor) -> Tensor:
    """Whole-sequence embedding, L2-normalised. ``landmarks``: [B,T,106,2] raw coords."""
    return F.normalize(model(front_end(landmarks)), p=2, dim=-1)
