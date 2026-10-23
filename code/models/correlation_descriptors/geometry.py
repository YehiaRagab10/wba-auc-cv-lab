"""§1.3 Canonical geometric front-end — raw variant (no PCA).

Converts raw ``[B, T, 106, 2]`` pixel-coordinate landmarks into ``[B, 843, T]``
features on GPU, fully differentiable and vectorised.

Base channels (281, x3 with Δ/ΔΔ = 843):

     6  pose              log-scale, roll, tx, ty, yaw proxy, pitch proxy
   212  raw deformation   aligned shape minus clip-median shape, flattened 106x2
    42  region            7 regions x (centroid 2, log-area, sqrt-eigvals 2, orientation)
    21  inter-region      distances between the 7 region centroids

This deliberately replaces plan §1.3 step 5 (PCA expression coefficients, K=32)
with the raw aligned deformation — plan §2 ablation 4, "replace PCA+region by raw
aligned 212-dim deformation", except that the region and distance channels are
kept since they are not PCA-derived. Nothing here is fitted on the training set,
so there is no basis to store, load or keep in sync with a checkpoint.

Variable-length clips are handled by bucketing sequences of *equal* length
(§1.6) rather than by padding and masking: 98% of clips are exactly 100 frames,
so exact-length buckets are both cheap and exact, and every clip-level statistic
below (mean centroid, median IOD, median shape, Procrustes reference) stays
unbiased without mask bookkeeping.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LANDMARK_REGIONS: dict[str, list[int]] = {
    "face_contour": list(range(0, 33)),
    "left_eye": list(range(33, 43)),
    "left_eyebrow": list(range(43, 52)),
    "mouth": list(range(52, 72)),
    "nose": list(range(72, 87)),
    "right_eye": list(range(87, 97)),
    "right_eyebrow": list(range(97, 106)),
}
REGION_NAMES = list(LANDMARK_REGIONS)

LEFT_EYE_IDX = list(range(33, 43))
RIGHT_EYE_IDX = list(range(87, 97))
NOSE_IDX = list(range(72, 87))

N_POSE, N_DEFORM, N_REGION, N_DIST = 6, 212, 42, 21
# Channel slices into the 281-dim base vector, in concatenation order.
SLICE_POSE = slice(0, N_POSE)
SLICE_DEFORM = slice(N_POSE, N_POSE + N_DEFORM)
SLICE_REGION = slice(N_POSE + N_DEFORM, N_POSE + N_DEFORM + N_REGION)
SLICE_DIST = slice(N_POSE + N_DEFORM + N_REGION, N_POSE + N_DEFORM + N_REGION + N_DIST)
N_BASE = N_POSE + N_DEFORM + N_REGION + N_DIST  # 281
N_FEATURES = 3 * N_BASE                          # 843


def _shoelace_area(pts: Tensor) -> Tensor:
    """Absolute shoelace area of each point set, treated as a polygon. [*,N,2] -> [*]"""
    x, y = pts[..., 0], pts[..., 1]
    x1, y1 = torch.roll(x, -1, dims=-1), torch.roll(y, -1, dims=-1)
    return 0.5 * (x * y1 - x1 * y).sum(dim=-1).abs()


class GeometricFrontEnd(nn.Module):
    """Raw landmarks -> [B, 843, T] kinematic features. No fitted parameters."""

    def __init__(self, derivatives: bool = True, median_shape: bool = True,
                 pose: bool = True) -> None:
        super().__init__()
        self.derivatives = derivatives
        self.median_shape = median_shape   # ablation: static-shape leakage
        self.use_pose = pose               # ablation: pose channels on/off
        # Region membership as flat maps, so every region descriptor below is a
        # scatter over all 106 landmarks at once rather than a loop over 7 regions
        # (which cost 7 separate median sorts and dominated the front-end).
        region_of = torch.zeros(106, dtype=torch.long)
        next_in_region = torch.zeros(106, dtype=torch.long)
        for r, (name, indices) in enumerate(LANDMARK_REGIONS.items()):
            idx = torch.tensor(indices, dtype=torch.long)
            region_of[idx] = r
            next_in_region[idx] = idx.roll(-1)      # cyclic successor, for the shoelace sum
            self.register_buffer(f"_idx_{name}", idx)
        self.register_buffer("_region_of", region_of)
        self.register_buffer("_next_in_region", next_in_region)
        self.register_buffer("_region_count",
                             torch.bincount(region_of, minlength=len(REGION_NAMES)).float())
        pairs = [(i, j) for i in range(len(REGION_NAMES)) for j in range(i + 1, len(REGION_NAMES))]
        self.register_buffer("_region_pairs", torch.tensor(pairs, dtype=torch.long))
        # 5-tap Savitzky-Golay first-derivative kernel: smoother than a raw diff,
        # which matters because landmark detections jitter frame to frame.
        self.register_buffer("_diff_kernel", torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]) / 10.0)

    @property
    def out_channels(self) -> int:
        n = N_BASE if self.use_pose else N_BASE - N_POSE
        return 3 * n if self.derivatives else n

    # ------------------------------------------------------------------ steps
    def _clip_normalise(self, x: Tensor) -> Tensor:
        """Step 1: subtract clip-mean centroid, divide by clip-median IOD.

        Clip-level (not per-frame) so head roll, nodding and approach/retreat survive.
        """
        x = x - x.mean(dim=(1, 2), keepdim=True)
        left = x[:, :, LEFT_EYE_IDX, :].mean(dim=2)
        right = x[:, :, RIGHT_EYE_IDX, :].mean(dim=2)
        iod = (left - right).norm(dim=-1).median(dim=1, keepdim=True).values
        return x / iod.clamp_min(1e-6)[..., None, None]

    def _procrustes(self, x: Tensor, n_iter: int = 2) -> tuple[Tensor, Tensor]:
        """Step 2: per-frame similarity Procrustes to the clip-mean shape.

        Closed form over complex coordinates: with z_t = x + iy, the similarity that
        best maps frame t onto the reference is a_t = <z_ref, z_t> / <z_t, z_t>.
        Two iterations of reference refinement.
        """
        z = torch.complex(x[..., 0], x[..., 1])          # [B,T,106]
        ref = z.mean(dim=1, keepdim=True)
        z_al, a_t = z, None
        for _ in range(n_iter):
            num = (ref.conj() * z).sum(dim=-1)
            den = (z * z.conj()).real.sum(dim=-1)
            a_t = num / den.clamp_min(1e-10)
            z_al = a_t.unsqueeze(-1) * z
            ref = z_al.mean(dim=1, keepdim=True)
        return torch.stack([z_al.real, z_al.imag], dim=-1), a_t

    def _pose(self, a_t: Tensor, x_norm: Tensor) -> Tensor:
        """Step 3: 6 pose channels, including out-of-plane proxies Procrustes cannot remove."""
        log_scale = a_t.abs().clamp_min(1e-10).log()
        roll = a_t.angle()
        centroid = x_norm.mean(dim=2)
        centroid = centroid - centroid.mean(dim=1, keepdim=True)
        tx, ty = centroid[..., 0], centroid[..., 1]

        nose = x_norm[:, :, NOSE_IDX, :].mean(dim=2)
        # Contour halves give a left/right asymmetry that grows with yaw. x_norm is
        # already in IOD units, so no further division is needed.
        d_left = (nose - x_norm[:, :, :17, :].mean(dim=2)).norm(dim=-1)
        d_right = (nose - x_norm[:, :, 17:33, :].mean(dim=2)).norm(dim=-1)
        yaw = d_left - d_right

        eye_y = 0.5 * (x_norm[:, :, LEFT_EYE_IDX, 1].mean(dim=2)
                       + x_norm[:, :, RIGHT_EYE_IDX, 1].mean(dim=2))
        height = (x_norm[..., 1].max(dim=2).values - x_norm[..., 1].min(dim=2).values).clamp_min(1e-6)
        pitch = (nose[..., 1] - eye_y) / height

        out = torch.stack([log_scale, roll, tx, ty, yaw, pitch], dim=-1)   # [B,T,6]
        # Median-centre the absolute tracks so resting pose (an avatar property)
        # drops out and only the dynamics remain. tx/ty are already centred.
        keep = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], device=out.device)
        return out - out.median(dim=1, keepdim=True).values * keep

    def _scatter(self, vals: Tensor) -> Tensor:
        """Sum per-landmark values into their regions. [B,T,106] -> [B,T,7]"""
        b, t = vals.shape[:2]
        out = vals.new_zeros(b, t, len(REGION_NAMES))
        return out.index_add_(2, self._region_of, vals)

    def _region_centroids(self, aligned: Tensor) -> Tensor:
        """Mean position of each region. [B,T,106,2] -> [B,T,7,2]"""
        sx = self._scatter(aligned[..., 0])
        sy = self._scatter(aligned[..., 1])
        return torch.stack([sx, sy], dim=-1) / self._region_count[:, None]

    def _regions(self, aligned: Tensor, cents: Tensor) -> Tensor:
        """Step 6: 6 descriptors per region, clip-median removed. -> [B,T,42]

        Computed for all 7 regions simultaneously by scattering per-landmark
        quantities into region bins, with a single median call at the end.
        """
        x, y = aligned[..., 0], aligned[..., 1]

        # Shoelace area, using the precomputed cyclic successor within each region.
        xn, yn = x[:, :, self._next_in_region], y[:, :, self._next_in_region]
        area = (0.5 * self._scatter(x * yn - xn * y)).abs().clamp_min(1e-10).log()

        # 2x2 covariance per region from centred coordinates.
        cx = x - cents[..., 0][:, :, self._region_of]
        cy = y - cents[..., 1][:, :, self._region_of]
        denom = (self._region_count - 1).clamp_min(1)
        a = self._scatter(cx * cx) / denom
        b = self._scatter(cx * cy) / denom
        d = self._scatter(cy * cy) / denom

        half_tr = (a + d) / 2
        disc = (half_tr.square() - (a * d - b * b).clamp_min(0)).clamp_min(0).sqrt()
        e1 = (half_tr + disc).clamp_min(0).sqrt()
        e2 = (half_tr - disc).clamp_min(0).sqrt()
        orient = torch.atan2(b, a - d + disc.clamp_min(1e-10))

        # Interleave to 7 regions x 6 descriptors, region-major.
        out = torch.stack([cents[..., 0], cents[..., 1], area, e1, e2, orient], dim=-1)
        out = out.reshape(*out.shape[:2], -1)                     # [B,T,42]
        return out - out.median(dim=1, keepdim=True).values

    def _distances(self, cents: Tensor) -> Tensor:
        """Step 7: 21 inter-region centroid distances, clip-median removed."""
        p = self._region_pairs
        d = (cents[:, :, p[:, 0], :] - cents[:, :, p[:, 1], :]).norm(dim=-1)
        return d - d.median(dim=1, keepdim=True).values

    def _derivatives(self, x: Tensor) -> Tensor:
        """Step 8: Δ and ΔΔ via a depthwise 5-tap conv. [B,C,T] -> [B,3C,T]"""
        c = x.shape[1]
        k = self._diff_kernel.to(x.dtype).view(1, 1, 5).expand(c, 1, 5)
        delta = F.conv1d(F.pad(x, (2, 2), mode="replicate"), k, groups=c)
        ddelta = F.conv1d(F.pad(delta, (2, 2), mode="replicate"), k, groups=c)
        return torch.cat([x, delta, ddelta], dim=1)

    # ---------------------------------------------------------------- forward
    def forward(self, x: Tensor) -> Tensor:
        """[B,T,106,2] raw pixel coords -> [B, out_channels, T]."""
        x_norm = self._clip_normalise(x.float())
        aligned, a_t = self._procrustes(x_norm)

        # Step 4: remove the clip-median shape, i.e. the target avatar static face.
        deform = aligned - aligned.median(dim=1, keepdim=True).values if self.median_shape else aligned
        cents = self._region_centroids(aligned)
        parts = []
        if self.use_pose:
            parts.append(self._pose(a_t, x_norm))
        parts += [deform.reshape(*deform.shape[:2], -1),
                  self._regions(aligned, cents), self._distances(cents)]

        base = torch.cat(parts, dim=-1).permute(0, 2, 1).contiguous()   # [B, 281, T]
        out = self._derivatives(base) if self.derivatives else base
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def handcrafted_channels(self) -> Tensor:
        """Indices of the M0 = 48 handcrafted DCS signals (§4.1): 6 pose + 42 region.

        Taken from the undifferentiated part of the base vector, so the correlations
        describe the signals themselves rather than their derivatives. With PCA
        excluded this is exactly 48 signals, so the handcrafted correlation block
        keeps the 1128 dimensions the plan specifies.
        """
        if not self.use_pose:
            return torch.arange(SLICE_REGION.start - N_POSE, SLICE_REGION.stop - N_POSE)
        return torch.cat([torch.arange(SLICE_POSE.start, SLICE_POSE.stop),
                          torch.arange(SLICE_REGION.start, SLICE_REGION.stop)])
