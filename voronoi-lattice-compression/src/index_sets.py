"""
Ensembles d'indices pour la méthode Gnewuch-Harsha-Wnuk (2024).

Reconstruit d'après les formules de l'article (Sections 3.5-3.6) et le
contrat d'appel de src/ghw_lattice.py :

- kappa_bound(gamma_j, threshold, alpha)
      borne kappa = floor((gamma_j * threshold)^(1/(2*alpha))) telle que
      r_alpha(gamma_j, k) <= threshold  <=>  |k| <= kappa ;

- rectangle_indices(nu, gamma, alpha)
      itere tous les multi-indices k du rectangle
      R^alpha_{nu,gamma,d} = {k : r_alpha(gamma_j, k_j) <= nu  pour tout j} ;

- rectangle_size(nu, gamma, alpha)
      cardinal du rectangle, prod_j (2*kappa_j + 1) ;

- step_hyperbolic_cross_decomposition(m, gamma, alpha)
      decomposition de la croix hyperbolique en escalier
      Q^alpha_{m,gamma,d} en blocs rectangulaires dyadiques indexes par les
      vecteurs de forme t in N_0^d, |t|_1 = m (formule (38) de GHW) :
        * coordonnee j = 0 : plage pleine |k_1| <= kappa_up_1,
          facteur D_{kappa_up_1} (convention lo_1 = -1) ;
        * coordonnees j >= 1 : coquille kappa_low_j < |k_j| <= kappa_up_j,
          facteur D_{kappa_up_j} - D_{kappa_low_j}, avec
          kappa_low_j = 0 si t_j = 0.
      Retourne une liste de paires (t, intervals) ou intervals[j] = (lo, up).
      Les blocs dont un facteur est identiquement nul (up == lo pour un
      j >= 1, ou up < 0) sont omis.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterator, List, Tuple

import numpy as np


def kappa_bound(gamma_j: float, threshold: float, alpha: int) -> int:
    """kappa = floor((gamma_j * threshold)^(1/(2*alpha))), >= 0.

    Caracterise r_alpha(gamma_j, k) = max(|k|^{2*alpha}/gamma_j, 1) <= threshold
    (pour threshold >= 1) par |k| <= kappa.
    """
    val = gamma_j * threshold
    if val < 1.0:
        return 0
    # garde numerique contre floor(0.99999...) sur les puissances exactes
    kappa = int(math.floor(val ** (1.0 / (2.0 * alpha)) + 1e-12))
    return max(kappa, 0)


def _shape_vectors(m: int, d: int) -> Iterator[Tuple[int, ...]]:
    """Tous les t in N_0^d avec |t|_1 = m (ordre lexicographique)."""
    if d == 1:
        yield (m,)
        return
    for t1 in range(m + 1):
        for rest in _shape_vectors(m - t1, d - 1):
            yield (t1,) + rest


def step_hyperbolic_cross_decomposition(
    m: int,
    gamma: np.ndarray,
    alpha: int,
) -> List[Tuple[Tuple[int, ...], List[Tuple[int, int]]]]:
    """Blocs rectangulaires dyadiques de Q^alpha_{m,gamma,d} (formule (38))."""
    gamma = np.asarray(gamma, dtype=np.float64)
    d = gamma.shape[0]
    blocks: List[Tuple[Tuple[int, ...], List[Tuple[int, int]]]] = []
    for t in _shape_vectors(m, d):
        intervals: List[Tuple[int, int]] = []
        empty = False
        for j in range(d):
            up = kappa_bound(float(gamma[j]), 2.0 ** t[j], alpha)
            if j == 0:
                lo = -1                      # convention D_{-1} = 0
            else:
                if t[j] == 0:
                    # niveau t_j = 0 : la coquille doit contenir k_j = 0
                    # (r_alpha(gamma_j, 0) = 1 <= 2^0) ; convention lo = -1,
                    # facteur D_up - D_{-1} = D_up. Prendre lo = 0 (comme la
                    # phrase de GHW apres (38)) exclurait l'hyperplan
                    # {k_j = 0} de tous les blocs et briserait la
                    # decomposition (34)-(35).
                    lo = -1
                else:
                    lo = kappa_bound(float(gamma[j]), 2.0 ** (t[j] - 1), alpha)
                    if up <= lo:             # coquille vide -> facteur nul
                        empty = True
                        break
            intervals.append((lo, up))
        if not empty:
            blocks.append((t, intervals))
    return blocks


def rectangle_indices(
    nu: float,
    gamma: np.ndarray,
    alpha: int,
) -> Iterator[Tuple[int, ...]]:
    """Multi-indices du rectangle R^alpha_{nu,gamma,d} (Section 3.5 de GHW)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    kappas = [kappa_bound(float(g), nu, alpha) for g in gamma]
    ranges = [range(-k, k + 1) for k in kappas]
    return itertools.product(*ranges)


def rectangle_size(nu: float, gamma: np.ndarray, alpha: int) -> int:
    """Cardinal du rectangle : prod_j (2*kappa_j + 1)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    size = 1
    for g in gamma:
        size *= 2 * kappa_bound(float(g), nu, alpha) + 1
    return size
