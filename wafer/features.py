"""Hand-crafted spatial features for the classical baseline.

Mirrors the feature engineering used before deep learning in wafer-map analysis:
where on the wafer the defective dies are (rings x sectors), how dense they are,
and the shape of the largest defect cluster (scratches are long and thin).
"""
import numpy as np
from scipy import ndimage

N_RINGS, N_SECTORS = 5, 8
_yy, _xx = np.mgrid[0:52, 0:52]
_R = np.hypot(_yy - 25.5, _xx - 25.5)
_THETA = (np.arctan2(_yy - 25.5, _xx - 25.5) + np.pi) / (2 * np.pi)  # 0..1


def wafer_features(x):
    wafer = x > 0
    defect = x == 2
    r = _R / max(_R[wafer].max(), 1)                      # radius normalised to the wafer edge
    ring = np.minimum((r * N_RINGS).astype(int), N_RINGS - 1)
    sector = np.minimum((_THETA * N_SECTORS).astype(int), N_SECTORS - 1)

    feats = [defect.sum() / max(wafer.sum(), 1)]           # overall defect ratio
    ring_density = []
    for i in range(N_RINGS):
        m = wafer & (ring == i)
        ring_density.append(defect[m].mean() if m.any() else 0.0)
    feats += ring_density
    grid = []
    for i in range(N_RINGS):
        for j in range(N_SECTORS):
            m = wafer & (ring == i) & (sector == j)
            grid.append(defect[m].mean() if m.any() else 0.0)
    feats += grid
    # How unevenly the defects are spread around the wafer (localised vs ring-like)
    outer = np.array(grid[-N_SECTORS:])
    feats += [outer.std(), outer.max() - outer.min()]

    labels, n = ndimage.label(defect)
    feats.append(n)
    if n:
        sizes = ndimage.sum(defect, labels, range(1, n + 1))
        big = labels == (np.argmax(sizes) + 1)
        ys, xs = np.nonzero(big)
        cov = np.cov(np.vstack([ys, xs])) if len(ys) > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))
        feats += [sizes.max() / max(wafer.sum(), 1),       # largest cluster size
                  np.sqrt(ev[1] / max(ev[0], 1e-6)),       # elongation (scratches are long)
                  r[big].mean()]                           # where the main cluster sits
    else:
        feats += [0.0, 0.0, 0.0]
    return np.array(feats, dtype=np.float32)


def feature_matrix(X):
    return np.stack([wafer_features(x) for x in X])
