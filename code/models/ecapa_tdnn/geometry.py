"""Kinematic geometric front-end for facial landmarks.

Converts raw [B, T, 106, 2] pixel-coordinate landmarks into a compact
[B, 303, T] feature tensor on GPU, fully differentiable and vectorised.

Channels (101 base × 3 with Δ/ΔΔ = 303):
    6   pose (log-scale, roll, tx, ty, yaw proxy, pitch proxy)
   32   PCA expression coefficients
   42   region descriptors (7 regions × 6: centroid, log-area, sqrt-eigvals, orient)
   21   inter-region centroid distances (7-choose-2)

Also provides a ``RawFrontEnd`` that simply flattens 106×2 + Δ/ΔΔ = 636 channels,
for smoke-testing and as a permanent ablation baseline.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Landmark region definitions (InsightFace 2D-106)
# ---------------------------------------------------------------------------

LANDMARK_REGIONS: dict[str, list[int]] = {
    "face_contour": list(range(0, 33)),
    "left_eye": list(range(33, 43)),
    "left_eyebrow": list(range(43, 52)),
    "mouth": list(range(52, 72)),
    "nose": list(range(72, 87)),
    "right_eye": list(range(87, 97)),
    "right_eyebrow": list(range(97, 106)),
}

REGION_NAMES = list(LANDMARK_REGIONS.keys())

LEFT_EYE_IDX = list(range(33, 43))
RIGHT_EYE_IDX = list(range(87, 97))
NOSE_IDX = list(range(72, 87))
CONTOUR_IDX = list(range(0, 33))


# ---------------------------------------------------------------------------
# Smoothed derivative kernel (5-tap)
# ---------------------------------------------------------------------------

def _make_diff_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
    """5-tap Savitzky-Golay first-derivative kernel, replicate-padded."""
    # Coefficients for 5-point SG first derivative: [-2, -1, 0, 1, 2] / 10
    k = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device, dtype=dtype) / 10.0
    return k


# ---------------------------------------------------------------------------
# Helper: shoelace area
# ---------------------------------------------------------------------------

def _shoelace_area(pts: Tensor) -> Tensor:
    """Signed shoelace area for a set of 2D points (treats them as a polygon).

    Args:
        pts: [*, N, 2]

    Returns:
        [*] area (absolute value)
    """
    x, y = pts[..., 0], pts[..., 1]
    # Close the polygon: shift by 1
    x1 = torch.roll(x, -1, dims=-1)
    y1 = torch.roll(y, -1, dims=-1)
    area = 0.5 * (x * y1 - x1 * y).sum(dim=-1).abs()
    return area


# ---------------------------------------------------------------------------
# KinematicFrontEnd
# ---------------------------------------------------------------------------

class KinematicFrontEnd(nn.Module):
    """GPU module that converts raw landmarks to 303-channel kinematic features.

    Input:  x [B, T, 106, 2] in raw pixel coordinates
    Output: [B, 303, T]

    Must call ``load_pca(basis, mean)`` before first forward with the full
    front-end. The PCA basis is saved as a buffer and serialised in the
    checkpoint.
    """

    def __init__(self, pca_k: int = 32) -> None:
        super().__init__()
        self.pca_k = pca_k
        # PCA buffers — initialised to None; call load_pca() before use
        self.register_buffer("pca_basis", torch.zeros(pca_k, 212))
        self.register_buffer("pca_mean", torch.zeros(212))
        self._pca_loaded = False

        # Pre-compute region index tensors as buffers
        for name, indices in LANDMARK_REGIONS.items():
            self.register_buffer(
                f"_idx_{name}",
                torch.tensor(indices, dtype=torch.long),
            )

        # All pairs of 7 regions for inter-region distances
        pairs = []
        for i in range(len(REGION_NAMES)):
            for j in range(i + 1, len(REGION_NAMES)):
                pairs.append((i, j))
        self.register_buffer("_region_pairs", torch.tensor(pairs, dtype=torch.long))

        # 5-tap diff kernel
        self.register_buffer(
            "_diff_kernel",
            torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]) / 10.0,
        )

    def load_pca(self, basis: Tensor, mean: Tensor) -> None:
        """Load PCA basis [K, 212] and mean [212] from fit_pca.py."""
        assert basis.shape == (self.pca_k, 212), f"Expected ({self.pca_k}, 212), got {basis.shape}"
        assert mean.shape == (212,), f"Expected (212,), got {mean.shape}"
        self.pca_basis.copy_(basis)
        self.pca_mean.copy_(mean)
        self._pca_loaded = True

    def _clip_normalise(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Subtract clip-mean centroid, divide by clip-median IOD.

        Returns:
            normalised [B, T, 106, 2], iod [B, 1, 1, 1]
        """
        # Clip-mean centroid
        centroid = x.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1, 2]
        x = x - centroid

        # Clip-median IOD
        left_eye_c = x[:, :, LEFT_EYE_IDX, :].mean(dim=2)   # [B, T, 2]
        right_eye_c = x[:, :, RIGHT_EYE_IDX, :].mean(dim=2) # [B, T, 2]
        iod_per_frame = (left_eye_c - right_eye_c).norm(dim=-1)  # [B, T]
        iod = iod_per_frame.median(dim=1, keepdim=True).values   # [B, 1]
        iod = iod.clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)    # [B, 1, 1, 1]
        x = x / iod
        return x, iod

    def _procrustes_align(
        self, x: Tensor, n_iter: int = 2
    ) -> tuple[Tensor, Tensor]:
        """Per-frame similarity Procrustes to the clip-mean shape.

        Args:
            x: [B, T, 106, 2] clip-normalised landmarks

        Returns:
            aligned [B, T, 106, 2], complex scale-rotation a_t [B, T]
        """
        z = torch.complex(x[..., 0], x[..., 1])  # [B, T, 106]

        # Initial reference = clip mean
        ref = z.mean(dim=1, keepdim=True)  # [B, 1, 106]

        for _ in range(n_iter):
            # a_t = sum(conj(z_ref) * z_t) / sum(|z_t|^2)
            numerator = (ref.conj() * z).sum(dim=-1)   # [B, T]
            denominator = (z * z.conj()).real.sum(dim=-1)  # [B, T]
            a_t = numerator / denominator.clamp_min(1e-10)  # [B, T]

            # Align
            aligned_z = a_t.unsqueeze(-1) * z  # [B, T, 106]

            # Recompute reference
            ref = aligned_z.mean(dim=1, keepdim=True)

        aligned = torch.stack([aligned_z.real, aligned_z.imag], dim=-1)  # [B, T, 106, 2]
        return aligned, a_t

    def _pose_channels(
        self, a_t: Tensor, x_norm: Tensor, iod: Tensor
    ) -> Tensor:
        """Extract 6 pose channels.

        Args:
            a_t: [B, T] complex Procrustes scale-rotation
            x_norm: [B, T, 106, 2] clip-normalised (pre-Procrustes) landmarks
            iod: [B, 1, 1, 1]

        Returns:
            [B, T, 6]
        """
        B, T = a_t.shape

        # log|a_t| (scale ~ depth), angle(a_t) (roll)
        log_scale = a_t.abs().clamp_min(1e-10).log()       # [B, T]
        roll = a_t.angle()                                   # [B, T]

        # Centroid tx, ty (per-frame, minus clip mean)
        centroid = x_norm.mean(dim=2)                        # [B, T, 2]
        centroid = centroid - centroid.mean(dim=1, keepdim=True)
        tx, ty = centroid[..., 0], centroid[..., 1]          # [B, T] each

        # Yaw proxy: (dist nose-to-left-contour - dist nose-to-right-contour) / IOD
        nose_c = x_norm[:, :, NOSE_IDX, :].mean(dim=2)      # [B, T, 2]
        left_contour = x_norm[:, :, :17, :].mean(dim=2)      # [B, T, 2]  (left half)
        right_contour = x_norm[:, :, 17:33, :].mean(dim=2)   # [B, T, 2] (right half)
        d_left = (nose_c - left_contour).norm(dim=-1)
        d_right = (nose_c - right_contour).norm(dim=-1)
        iod_flat = iod.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1]
        # iod is already divided out in x_norm, so use 1.0 as reference
        yaw = d_left - d_right  # [B, T]

        # Pitch proxy: (nose centroid y - eye-line y) / face height
        eye_line_y = 0.5 * (
            x_norm[:, :, LEFT_EYE_IDX, 1].mean(dim=2)
            + x_norm[:, :, RIGHT_EYE_IDX, 1].mean(dim=2)
        )  # [B, T]
        nose_y = nose_c[..., 1]
        face_top = x_norm[:, :, :, 1].min(dim=2).values
        face_bot = x_norm[:, :, :, 1].max(dim=2).values
        face_height = (face_bot - face_top).clamp_min(1e-6)
        pitch = (nose_y - eye_line_y) / face_height  # [B, T]

        # Subtract clip medians from all pose tracks (removes static face shape)
        log_scale = log_scale - log_scale.median(dim=1, keepdim=True).values
        roll = roll - roll.median(dim=1, keepdim=True).values
        yaw = yaw - yaw.median(dim=1, keepdim=True).values
        pitch = pitch - pitch.median(dim=1, keepdim=True).values

        return torch.stack([log_scale, roll, tx, ty, yaw, pitch], dim=-1)  # [B, T, 6]

    def _deformation(self, aligned: Tensor) -> Tensor:
        """D_t = aligned_t - median_t(aligned). Removes static face shape.

        Returns:
            [B, T, 106, 2]
        """
        median_shape = aligned.median(dim=1, keepdim=True).values
        return aligned - median_shape

    def _pca_expression(self, deformation: Tensor) -> Tensor:
        """Project flattened deformation onto PCA basis.

        Args:
            deformation: [B, T, 106, 2]

        Returns:
            [B, T, K]
        """
        B, T = deformation.shape[:2]
        flat = deformation.reshape(B, T, -1)  # [B, T, 212]
        centered = flat - self.pca_mean.unsqueeze(0).unsqueeze(0)
        # Project: [B, T, 212] @ [212, K] -> [B, T, K]
        coeffs = centered @ self.pca_basis.T
        return coeffs

    def _region_descriptors(self, aligned: Tensor) -> Tensor:
        """6 descriptors per region (centroid, log-area, sqrt-eigvals, orientation).

        Args:
            aligned: [B, T, 106, 2]

        Returns:
            [B, T, 42]
        """
        B, T = aligned.shape[:2]
        descs = []

        for name in REGION_NAMES:
            idx = getattr(self, f"_idx_{name}")
            pts = aligned[:, :, idx, :]  # [B, T, N_region, 2]

            # Centroid
            c = pts.mean(dim=2)  # [B, T, 2]

            # Log shoelace area
            area = _shoelace_area(pts).clamp_min(1e-10).log()  # [B, T]

            # 2x2 covariance eigenvalues and orientation
            centered = pts - c.unsqueeze(2)  # [B, T, N, 2]
            # Covariance: [B, T, 2, 2]
            cov = torch.einsum("btni,btnj->btij", centered, centered) / max(
                1, pts.shape[2] - 1
            )
            # Eigenvalues via closed-form 2x2
            a, b, d = cov[:, :, 0, 0], cov[:, :, 0, 1], cov[:, :, 1, 1]
            trace = a + d
            det = (a * d - b * b).clamp_min(0)
            disc = ((trace / 2).square() - det).clamp_min(0).sqrt()
            e1 = (trace / 2 + disc).clamp_min(0).sqrt()  # [B, T]
            e2 = (trace / 2 - disc).clamp_min(0).sqrt()  # [B, T]

            # Orientation: angle of first eigenvector
            orient = torch.atan2(b, a - d + disc.clamp_min(1e-10))  # [B, T]

            descs.append(torch.stack([c[..., 0], c[..., 1], area, e1, e2, orient], dim=-1))

        result = torch.cat(descs, dim=-1)  # [B, T, 42]

        # Subtract clip median
        clip_median = result.median(dim=1, keepdim=True).values
        return result - clip_median

    def _inter_region_distances(self, aligned: Tensor) -> Tensor:
        """Pairwise distances between 7 region centroids, minus clip median.

        Returns:
            [B, T, 21]
        """
        B, T = aligned.shape[:2]
        centroids = []
        for name in REGION_NAMES:
            idx = getattr(self, f"_idx_{name}")
            centroids.append(aligned[:, :, idx, :].mean(dim=2))  # [B, T, 2]
        centroids = torch.stack(centroids, dim=2)  # [B, T, 7, 2]

        # All pairs
        pairs = self._region_pairs  # [21, 2]
        c_i = centroids[:, :, pairs[:, 0], :]  # [B, T, 21, 2]
        c_j = centroids[:, :, pairs[:, 1], :]  # [B, T, 21, 2]
        dists = (c_i - c_j).norm(dim=-1)       # [B, T, 21]

        clip_median = dists.median(dim=1, keepdim=True).values
        return dists - clip_median

    def _compute_derivatives(self, x: Tensor) -> Tensor:
        """Compute Δ and ΔΔ using 5-tap smoothed difference via F.conv1d.

        Args:
            x: [B, C, T]

        Returns:
            [B, 3C, T]  (original, Δ, ΔΔ)
        """
        B, C, T = x.shape
        kernel = self._diff_kernel  # [5]

        # Reshape for depthwise conv: kernel [C, 1, 5]
        k = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, 5)

        # Replicate-pad
        padded = F.pad(x, (2, 2), mode="replicate")
        delta = F.conv1d(padded, k, groups=C)  # [B, C, T]

        # ΔΔ
        padded_d = F.pad(delta, (2, 2), mode="replicate")
        delta_delta = F.conv1d(padded_d, k, groups=C)  # [B, C, T]

        return torch.cat([x, delta, delta_delta], dim=1)  # [B, 3C, T]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, 106, 2] raw pixel coordinates

        Returns:
            [B, 303, T]
        """
        # 1. Clip-level normalisation
        x_norm, iod = self._clip_normalise(x)

        # 2. Procrustes alignment
        aligned, a_t = self._procrustes_align(x_norm)

        # 3. Pose (6 channels)
        pose = self._pose_channels(a_t, x_norm, iod)  # [B, T, 6]

        # 4. Deformation
        deformation = self._deformation(aligned)  # [B, T, 106, 2]

        # 5. PCA expression (32 channels)
        pca_expr = self._pca_expression(deformation)  # [B, T, 32]

        # 6. Region descriptors (42 channels)
        region_desc = self._region_descriptors(aligned)  # [B, T, 42]

        # 7. Inter-region distances (21 channels)
        inter_dist = self._inter_region_distances(aligned)  # [B, T, 21]

        # Concatenate base features: 6 + 32 + 42 + 21 = 101
        base = torch.cat([pose, pca_expr, region_desc, inter_dist], dim=-1)  # [B, T, 101]

        # Transpose to [B, 101, T] for conv1d derivatives
        base = base.permute(0, 2, 1).contiguous()

        # 8. Derivatives: [B, 303, T]
        features = self._compute_derivatives(base)

        # Clean up
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features


# ---------------------------------------------------------------------------
# Raw front-end (ablation baseline)
# ---------------------------------------------------------------------------

class RawFrontEnd(nn.Module):
    """Flatten 106×2 landmarks + Δ/ΔΔ = 636 channels. No geometric processing."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_diff_kernel",
            torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]) / 10.0,
        )

    def _clip_normalise(self, x: Tensor) -> Tensor:
        """Minimal normalisation: subtract clip-mean centroid, divide by IOD."""
        centroid = x.mean(dim=(1, 2), keepdim=True)
        x = x - centroid
        left_c = x[:, :, LEFT_EYE_IDX, :].mean(dim=2)
        right_c = x[:, :, RIGHT_EYE_IDX, :].mean(dim=2)
        iod = (left_c - right_c).norm(dim=-1).median(dim=1, keepdim=True).values
        iod = iod.clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)
        return x / iod

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, 106, 2]

        Returns:
            [B, 636, T]
        """
        x = self._clip_normalise(x)
        B, T = x.shape[:2]
        flat = x.reshape(B, T, -1)  # [B, T, 212]
        flat = flat.permute(0, 2, 1).contiguous()  # [B, 212, T]

        # Derivatives
        C = flat.shape[1]
        kernel = self._diff_kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, 5)
        padded = F.pad(flat, (2, 2), mode="replicate")
        delta = F.conv1d(padded, kernel, groups=C)
        padded_d = F.pad(delta, (2, 2), mode="replicate")
        delta_delta = F.conv1d(padded_d, kernel, groups=C)

        features = torch.cat([flat, delta, delta_delta], dim=1)  # [B, 636, T]
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# GPU augmentations (applied to raw landmarks before front-end, training only)
# ---------------------------------------------------------------------------

class LandmarkAugmentation(nn.Module):
    """GPU-based augmentations on raw [B, T, 106, 2] landmarks.

    Applied during training before the front-end.
    """

    def __init__(
        self,
        expression_gain_range: tuple[float, float] = (0.7, 1.2),
        smooth_sigma_max: float = 1.5,
        jitter_frac: tuple[float, float] = (0.005, 0.02),
        speed_range: tuple[float, float] = (0.9, 1.1),
        frame_drop_rate: float = 0.05,
        roll_deg: float = 10.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
    ) -> None:
        super().__init__()
        self.expression_gain_range = expression_gain_range
        self.smooth_sigma_max = smooth_sigma_max
        self.jitter_frac = jitter_frac
        self.speed_range = speed_range
        self.frame_drop_rate = frame_drop_rate
        self.roll_deg = roll_deg
        self.scale_range = scale_range

        # Region indices for per-region expression gain
        self._region_slices: list[Tensor] = []
        for name, indices in LANDMARK_REGIONS.items():
            self.register_buffer(
                f"_aug_idx_{name}",
                torch.tensor(indices, dtype=torch.long),
            )

    def _global_similarity(self, x: Tensor) -> Tensor:
        """Random global roll and scale."""
        angle = random.uniform(-self.roll_deg, self.roll_deg)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        scale = random.uniform(*self.scale_range)
        rot = torch.tensor(
            [[cos_a * scale, -sin_a * scale], [sin_a * scale, cos_a * scale]],
            device=x.device, dtype=x.dtype,
        )
        centroid = x.mean(dim=(1, 2), keepdim=True)
        x = (x - centroid) @ rot.T + centroid
        return x

    def _expression_gain(self, x: Tensor) -> Tensor:
        """Per-region expression gain on the deformation."""
        # Compute clip-median shape
        median_shape = x.median(dim=1, keepdim=True).values  # [B, 1, 106, 2]
        D = x - median_shape

        for name in LANDMARK_REGIONS:
            idx = getattr(self, f"_aug_idx_{name}")
            gain = random.uniform(*self.expression_gain_range)
            D[:, :, idx, :] = D[:, :, idx, :] * gain

        return median_shape + D

    def _temporal_smooth(self, x: Tensor) -> Tensor:
        """Gaussian temporal smoothing."""
        sigma = random.uniform(0, self.smooth_sigma_max)
        if sigma < 0.1:
            return x

        B, T, V, C = x.shape
        # Build Gaussian kernel
        ks = max(3, int(4 * sigma + 1)) | 1  # ensure odd
        half = ks // 2
        t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (t / sigma).square())
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, ks]

        # Apply per-channel: reshape to [B*V*C, 1, T]
        flat = x.permute(0, 2, 3, 1).reshape(B * V * C, 1, T)
        padded = F.pad(flat, (half, half), mode="replicate")
        smoothed = F.conv1d(padded, kernel)
        return smoothed.reshape(B, V, C, T).permute(0, 3, 1, 2)

    def _jitter(self, x: Tensor) -> Tensor:
        """Gaussian landmark jitter scaled by IOD."""
        left_c = x[:, :, LEFT_EYE_IDX, :].mean(dim=2)
        right_c = x[:, :, RIGHT_EYE_IDX, :].mean(dim=2)
        iod = (left_c - right_c).norm(dim=-1).mean(dim=1, keepdim=True)  # [B, 1]
        frac = random.uniform(*self.jitter_frac)
        noise_scale = iod.unsqueeze(-1).unsqueeze(-1) * frac  # [B, 1, 1, 1]
        return x + torch.randn_like(x) * noise_scale

    def _speed_perturb(self, x: Tensor) -> Tensor:
        """Resample at a random speed via linear interpolation."""
        B, T, V, C = x.shape
        if T < 2:
            return x
        speed = random.uniform(*self.speed_range)
        new_T = max(2, int(T / speed))
        indices = torch.linspace(0, T - 1, new_T, device=x.device)
        idx0 = indices.long().clamp(max=T - 2)
        idx1 = idx0 + 1
        frac = (indices - idx0.float()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, T', 1, 1]
        return x[:, idx0] * (1 - frac) + x[:, idx1] * frac

    def _frame_drop_duplicate(self, x: Tensor) -> Tensor:
        """Randomly drop or duplicate up to 5% of frames."""
        B, T, V, C = x.shape
        n_affected = max(0, int(T * self.frame_drop_rate))
        if n_affected == 0 or T < 4:
            return x
        indices = list(range(T))
        for _ in range(n_affected):
            if random.random() < 0.5:
                # Drop a frame (replace with neighbor)
                pos = random.randint(0, len(indices) - 1)
                indices[pos] = indices[max(0, pos - 1)]
            else:
                # Duplicate a frame
                pos = random.randint(0, len(indices) - 1)
                indices.insert(pos, indices[pos])
        # Truncate or pad back to T
        if len(indices) > T:
            indices = indices[:T]
        elif len(indices) < T:
            indices.extend([indices[-1]] * (T - len(indices)))
        idx_t = torch.tensor(indices, device=x.device, dtype=torch.long)
        return x[:, idx_t]

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, 106, 2]

        Returns:
            [B, T', 106, 2] (T' may differ slightly due to speed perturb)
        """
        x = self._global_similarity(x)
        x = self._expression_gain(x)
        x = self._temporal_smooth(x)
        x = self._jitter(x)
        x = self._speed_perturb(x)
        x = self._frame_drop_duplicate(x)
        return x