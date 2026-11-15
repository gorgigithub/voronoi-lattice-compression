"""
Construction de digital nets (low-discrepancy point sets).

Pour le prototype, on s'appuie sur scipy.stats.qmc qui fournit des
implémentations robustes de Sobol et Halton. À terme, pour reproduire
exactement les expériences de Dick & Feischl ou Göttlich et al., il
faudrait charger les matrices génératrices Niederreiter-Xing depuis
le "Magic Point Shop" de Nuyens :
    https://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/

Référence :
    Dick & Pillichshammer (2010), "Digital Nets and Sequences", CUP.

ATTENTION (piège du point central) :
    Le Sobol NON scramblé de scipy place son deuxième point exactement au
    centre (0.5, ..., 0.5) du cube. Pour une fonction très piquée au centre
    en haute dimension (par ex. f1(x) = exp(-a*sum|x_i - 0.5|)), ce point
    crée un cas pathologique : f(z)^2 = 1 avec un poids QMC-averaging trop
    grand, ce qui fait exploser l'erreur. Le scrambling élimine ce point et
    reproduit le comportement des Niederreiter-Xing utilisés par Göttlich
    et al. C'est pourquoi scramble=True est désormais le DÉFAUT.
    Pour une étude déterministe stricte (sans aléa), utiliser scramble=False
    en connaissance de cause.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from scipy.stats import qmc
    _HAS_SCIPY_QMC = True
except ImportError:
    _HAS_SCIPY_QMC = False


def sobol_points(L: int, d: int, seed: Optional[int] = None,
                 scramble: bool = True) -> np.ndarray:
    """
    Génère L points de la séquence de Sobol en dimension d.

    Pour avoir un (t, m, d)-net, prendre L = 2^m.

    Parameters
    ----------
    L : int
        Nombre de points. Idéalement L = 2^m pour la propriété de net.
    d : int
        Dimension.
    seed : int, optionnel
        Graine pour le scrambling. Fixer la graine garantit la
        reproductibilité tout en évitant le point central pathologique.
    scramble : bool
        DÉFAUT True (recommandé). Applique le scrambling d'Owen, ce qui
        élimine le point central (0.5, ..., 0.5) pathologique et reproduit
        le comportement des Niederreiter-Xing de Göttlich et al.
        Mettre False UNIQUEMENT pour une étude déterministe stricte, en
        sachant que cela réintroduit le piège du point central (voir le
        docstring du module).

    Returns
    -------
    P : np.ndarray, shape (L, d)
    """
    if not _HAS_SCIPY_QMC:
        raise ImportError("scipy.stats.qmc requis. pip install scipy>=1.7")

    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    # qmc.Sobol attend m tel que L = 2^m pour des garanties
    m = int(np.log2(L))
    if 2 ** m != L:
        # Sobol peut générer n'importe quel L, mais on perd la propriété de net
        P = sampler.random(L)
    else:
        P = sampler.random_base2(m)
    return np.asarray(P, dtype=np.float64)


def halton_points(L: int, d: int) -> np.ndarray:
    """
    Génère L points de la séquence de Halton.

    Utile comme baseline alternative.
    """
    if not _HAS_SCIPY_QMC:
        raise ImportError("scipy.stats.qmc requis.")
    sampler = qmc.Halton(d=d, scramble=False)
    return np.asarray(sampler.random(L), dtype=np.float64)


# -----------------------------------------------------------------------------
# Squelette pour digital nets construits via matrices génératrices
# -----------------------------------------------------------------------------

def digital_net_from_matrices(
    C: np.ndarray,
    m: int,
    b: int = 2,
) -> np.ndarray:
    """
    Construit un (t1, m, s)-net en base b à partir des matrices génératrices.

    Soit C de shape (s, m, m). Pour ell = 0, ..., b^m - 1, on décompose ell
    en base b : ell = sum_{i=0}^{m-1} lambda_i * b^i, puis
        z_{ell, j} = sum_{i=1}^{m} (C_j @ lambda)_i / b^i

    Parameters
    ----------
    C : np.ndarray, shape (s, m, m), dtype int
        Matrices génératrices (une par dimension).
    m : int
        log_b du nombre de points.
    b : int
        Base (premier).

    Returns
    -------
    P : np.ndarray, shape (b^m, s), dans [0, 1)^s
    """
    s = C.shape[0]
    L = b ** m
    P = np.zeros((L, s), dtype=np.float64)

    inv_powers = 1.0 / (b ** np.arange(1, m + 1))  # [1/b, 1/b^2, ...]

    for ell in range(L):
        # Décomposition en base b : lambda = (lambda_0, ..., lambda_{m-1})
        lam = np.zeros(m, dtype=np.int64)
        tmp = ell
        for i in range(m):
            lam[i] = tmp % b
            tmp //= b
        for j in range(s):
            # z_{ell,j,*} = C_j @ lambda mod b
            zj_digits = (C[j] @ lam) % b
            P[ell, j] = np.sum(zj_digits * inv_powers)

    return P


if __name__ == "__main__":
    print("Test Sobol (scramble=True par défaut) :")
    P = sobol_points(L=64, d=3, seed=0)
    print(f"  shape = {P.shape}, min = {P.min():.4f}, max = {P.max():.4f}")
    print(f"  premiers points :\n{P[:4]}")

    print("\nVérification : le point central n'est PLUS présent (scramble) :")
    P10 = sobol_points(L=256, d=10, seed=0)
    center = 0.5 * np.ones(10)
    dist_min = np.abs(P10 - center).sum(axis=1).min()
    print(f"  distance L1 min au centre = {dist_min:.4f} (doit être > 0)")

    print("\nComparaison : sans scramble, le centre EST présent (piège) :")
    P10_ns = sobol_points(L=256, d=10, seed=0, scramble=False)
    dist_min_ns = np.abs(P10_ns - center).sum(axis=1).min()
    print(f"  distance L1 min au centre = {dist_min_ns:.4f} (= 0 : pathologique)")

    print("\nTest Halton :")
    P = halton_points(L=64, d=3)
    print(f"  shape = {P.shape}")
