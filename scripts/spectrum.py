"""PCA basis of an embedding matrix, fitted on TRAIN and applied to any split.

Used to truncate the frozen backbone embedding to its informative subspace before
concatenating it with a fingerprint (``concat_balanced_molhiv.py``) and to sweep the
retained-dimension curve (``width_probe_molhiv.py``).

Fitting on TRAIN only is load-bearing: the rotation is part of the feature
extractor, so fitting it on train+test would leak the test distribution into the
representation and inflate the very transfer effect the concatenation measures.

``mode``:
  * ``raw``    - untouched (identity), for reference rows;
  * ``center`` - mean removal + rotation into the principal basis. Truncating to the
                 first k columns of this is the best k-dimensional linear summary
                 of the embedding, which is what the concat experiment uses;
  * ``beta``   - component i rescaled by lambda_i**((beta-1)/2), so its variance
                 becomes lambda_i**beta (beta=1 unchanged, beta=0 full whitening);
  * ``drop``   - centered with the top-k principal components deleted
                 ("all-but-the-top", the standard embedding-anisotropy fix).

Note ``beta`` and ``drop`` are not used by the molhiv concat results - a random
forest splits on per-feature thresholds and is therefore exactly invariant to any
positive per-axis rescaling, so ``beta`` cannot move an RF probe at all. They are
kept because the class is shared with cosine-similarity retrieval experiments,
where scale does bite.
"""

from __future__ import annotations

import numpy as np


class Spectrum:
    """PCA basis fitted on TRAIN embeddings; applies spectrum reshapings to any split."""

    def __init__(self, train_x: np.ndarray, var_floor: float = 1e-8):
        self.mean = train_x.mean(0, keepdims=True)
        xc = train_x - self.mean
        # SVD of the centered train matrix -> principal axes V, variances lam.
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        lam = (s ** 2) / max(len(xc) - 1, 1)
        keep = lam > var_floor * lam.max()      # drop numerically-dead directions
        self.V = vt[keep].T                     # (D, K)
        self.lam = lam[keep]                    # (K,)

    def apply(self, x: np.ndarray, mode: str, beta: float = 1.0, k: int = 0) -> np.ndarray:
        if mode == "raw":
            return x
        c = (x - self.mean) @ self.V            # centered PCA coordinates
        if mode == "center":
            return c
        if mode == "beta":
            # variance lambda_i -> lambda_i**beta  =>  scale by lambda_i**((beta-1)/2)
            return c * (self.lam ** ((beta - 1.0) / 2.0))[None, :]
        if mode == "drop":
            return c[:, k:]
        raise ValueError(f"unknown mode {mode!r}")
