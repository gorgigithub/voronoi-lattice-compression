"""
Version optimisée de compute_ghw_weights_step_hc (sortie identique).

Trois accélérations, sans changement mathématique :
1. les matrices diff_j = x_{.,j} - z_{.,j} (N x L) sont précalculées une
   seule fois par appel, au lieu d'une fois par bloc ;
2. les facteurs de Dirichlet sont mis en cache par clef (j, lo, up) et
   réutilisés entre les blocs (les paires distinctes sont peu nombreuses) ;
3. les facteurs identiquement égaux à 1 sont sautés : pour j >= 1,
   (lo, up) = (-1, 0) donne D_0 - D_{-1} = 1 ; pour j = 0, up = 0 donne
   D_0 = 1. En dimension élevée, la plupart des coordonnées d'un bloc
   sont dans ce cas, d'où l'essentiel du gain.
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple

# Plafond mémoire (Mo) du cache de facteurs de Dirichlet. À d=10, L=1031
# (m=20, 954 blocs) le cache complet pèse ~2.6 Go ; un plafond de 1500 Mo
# déclenche des évictions LRU avec un surcoût modeste (les blocs consécutifs
# de l'énumération lexicographique partagent la plupart de leurs facteurs).
# Augmentez à 3000+ si la machine dispose de >= 8 Go libres.
GHW_CACHE_MB = 1500

from src.index_sets import step_hyperbolic_cross_decomposition as _shcd

_DECOMP_CACHE: dict = {}

def step_hyperbolic_cross_decomposition(m, gamma, alpha):
    key = (m, tuple(np.asarray(gamma, dtype=float).tolist()), alpha)
    if key not in _DECOMP_CACHE:
        _DECOMP_CACHE[key] = _shcd(m, gamma, alpha)
    return _DECOMP_CACHE[key]
from src.ghw_lattice import dirichlet_kernel


def compute_ghw_weights_step_hc_fast(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    m: int,
    gamma: np.ndarray,
    alpha: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    N, d = X.shape
    L = Z.shape[0]
    assert Z.shape[1] == d

    decomposition = step_hyperbolic_cross_decomposition(m, gamma, alpha)

    # (1) diffs précalculées
    diffs = [X[:, j:j + 1] - Z[:, j].reshape(1, L) for j in range(d)]

    # (2) cache LRU des facteurs par (j, lo, up), plafonné en mémoire
    cache: "OrderedDict[Tuple[int, int, int], np.ndarray]" = OrderedDict()
    bytes_per = 8 * N * L
    max_entries = max(1, int(GHW_CACHE_MB * 1e6 / bytes_per))

    def factor(j: int, lo: int, up: int) -> np.ndarray:
        key = (j, lo, up)
        F = cache.get(key)
        if F is not None:
            cache.move_to_end(key)
            return F
        D_up = dirichlet_kernel(diffs[j], up)
        F = D_up if lo < 0 else D_up - dirichlet_kernel(diffs[j], lo)
        cache[key] = F
        if len(cache) > max_entries:
            cache.popitem(last=False)          # éviction LRU
        return F

    W_X = np.zeros(L, dtype=np.float64)
    W_XY = np.zeros(L, dtype=np.float64)

    for t, intervals in decomposition:
        contrib = None
        for j in range(d):
            lo, up = intervals[j]
            # (3) facteur trivialement == 1 : D_0 = 1 (j = 0, up = 0) ou
            #     D_0 - D_{-1} = 1 (j >= 1, lo = -1, up = 0)
            if up == 0 and lo < 0:
                continue
            F = factor(j, lo, up)
            contrib = F.copy() if contrib is None else contrib * F
        if contrib is None:                    # bloc entièrement trivial
            W_X += 1.0
            W_XY += float(Y.mean())
        else:
            W_X += contrib.sum(axis=0) / N
            W_XY += (Y[:, np.newaxis] * contrib).sum(axis=0) / N

    return W_X, W_XY
