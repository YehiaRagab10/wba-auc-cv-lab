"""§5 embedding back-end: centering -> LDA -> WCCN -> L2.

The organisers score submitted embeddings, so every back-end trick has to be an
embedding-level transform. This is the one implementation of it, used by the
classical baseline, by ``trainer.py`` at validation time, and by ``infer.py`` — plan
§1.6 note 4 exists because the old TCNN centred test embeddings in ``infer.py`` and
not in ``evaluate()``, and the two silently disagreed.

Why this matters here specifically: the classical variant of Approach C
(1176 handcrafted features -> LDA -> WCCN -> cosine) scores 0.7386 while the
learned network scores 0.5906 with a +0.26 held-in/held-out gap. The difference is
not the features, it is that LDA is a heavily shrunk linear projection and WCCN
explicitly whitens the directions that vary *within* a driver — the target
appearance and the speaking/listening mode. The network has no equivalent, so it
gets the same treatment applied to its embeddings.

Everything is fitted on train embeddings with train driver labels. No test or
validation statistics enter any transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def nap_directions(emb: np.ndarray, drivers: np.ndarray, sources: np.ndarray,
                   k: int = 8) -> np.ndarray:
    """Plan §5.3 - Nuisance Attribute Projection.

    Finds the directions along which the SAME driver moves when the TARGET AVATAR
    changes. Those directions encode the rendered appearance rather than the person,
    which is the nuisance a different-generator trial is swamped by. Returns the top-k
    such directions, to be projected out.

    Fitted on train embeddings with train labels only - no test or validation data.
    """
    diffs = []
    for d in np.unique(drivers):
        m = drivers == d
        e, sr = emb[m], sources[m]
        uniq = np.unique(sr)
        if len(uniq) < 2:
            continue
        # per-source means for this driver; their spread is the nuisance we want gone
        cent = np.stack([e[sr == u].mean(0) for u in uniq])
        diffs.append(cent - cent.mean(0, keepdims=True))
    if not diffs:
        return np.zeros((0, emb.shape[1]))
    D = np.concatenate(diffs, 0)
    # top-k principal directions of within-driver, between-source variation
    _u, _s, vt = np.linalg.svd(D - D.mean(0, keepdims=True), full_matrices=False)
    return vt[:k]


@dataclass
class Backend:
    """A fitted centering + LDA + WCCN transform, stored as plain arrays."""
    mean: np.ndarray
    scale: np.ndarray
    proj: np.ndarray            # [D, k] combined LDA scalings @ WCCN
    lda_mean: np.ndarray
    dims: int
    nap: np.ndarray | None = None      # [k, dims] directions to project out

    def apply(self, emb: Tensor) -> Tensor:
        x = emb.detach().float().cpu().numpy()
        z = (x - self.mean) / self.scale
        y = (z - self.lda_mean) @ self.proj
        if self.nap is not None and len(self.nap):
            y = y - (y @ self.nap.T) @ self.nap       # remove the nuisance subspace
        out = torch.from_numpy(y).float()
        return torch.nn.functional.normalize(out, p=2, dim=-1)

    def state(self) -> dict:
        return {"mean": self.mean, "scale": self.scale, "proj": self.proj,
                "lda_mean": self.lda_mean, "dims": self.dims}

    @staticmethod
    def from_state(s: dict) -> "Backend":
        return Backend(s["mean"], s["scale"], s["proj"], s["lda_mean"], int(s["dims"]))


def fit_backend(emb: Tensor, labels: np.ndarray, dims: int = 200,
                shrinkage: float = 0.1, wccn: bool = True,
                sources: np.ndarray | None = None, nap_k: int = 0) -> Backend | None:
    """Fit centering + LDA + WCCN on train embeddings.

    ``dims`` is capped at #classes - 1, which is what LDA can produce. Returns None
    if there are too few classes for LDA to mean anything.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    x = emb.detach().float().cpu().numpy()
    classes = np.unique(labels)
    if len(classes) < 3:
        return None

    mean = x.mean(0)
    scale = x.std(0) + 1e-6
    z = (x - mean) / scale

    k = min(dims, len(classes) - 1, z.shape[1])
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage, n_components=k)
    lda.fit(z, labels)
    scalings = lda.scalings_[:, :k]
    lda_mean = lda.means_.mean(axis=0)
    y = (z - lda_mean) @ scalings

    proj = scalings
    if wccn:
        # Within-class covariance: directions a single driver varies along are the
        # nuisance (target appearance, mode). Whitening by it down-weights them.
        w = np.zeros((y.shape[1], y.shape[1]))
        for c in classes:
            yc = y[labels == c]
            if len(yc) < 2:
                continue
            d = yc - yc.mean(0)
            w += d.T @ d
        w /= max(1, len(y) - len(classes))
        w += np.eye(len(w)) * 1e-4 * max(np.trace(w), 1e-12) / len(w)
        vals, vecs = np.linalg.eigh(w)
        whiten = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-10))) @ vecs.T
        proj = scalings @ whiten

    nap = None
    if nap_k and sources is not None:
        y2 = (z - lda_mean) @ proj
        nap = nap_directions(y2, labels, sources, nap_k)
        nap = nap / np.linalg.norm(nap, axis=1, keepdims=True).clip(1e-12)
    return Backend(mean=mean, scale=scale, proj=proj, lda_mean=lda_mean,
                   dims=proj.shape[1], nap=nap)
