"""
Méthode QMC-averaging pour la compression supervisée.

Implémente la méthode de Dick & Feischl (2021) avec digital nets
(formules (3.1)-(3.2) de Göttlich et al. 2025), et un squelette
pour la méthode de Gnewuch, Harsha, Wnuk (2024) avec lattices.

ATTENTION : ces méthodes ne renvoient PAS un couple (Z, W) où W est
une réponse compressée, mais (Z, weights_X, weights_XY) où les deux
ensembles de poids servent à approximer la perte sous la forme :

    appL(f) = sum_l f(z_l)^2 * W_{X,Z,nu,l}
            - 2 * sum_l f(z_l) * W_{X,Y,Z,nu,l}
            + (1/N) * sum_n y_n^2

cf. équation (1.3) de Göttlich et al. (2025).
"""

from __future__ import annotations

import numpy as np
from math import comb
from typing import Tuple


def _b_adic_expansion(n: int, b: int, m: int) -> np.ndarray:
    """Décomposition de n en base b sur m chiffres : n = sum lambda_i * b^i."""
    digits = np.zeros(m, dtype=np.int64)
    for i in range(m):
        digits[i] = n % b
        n //= b
    return digits


def _interval_index(x_coord: float, depth: int, b: int) -> int:
    """
    Pour x in [0, 1) et depth d_j, retourne l'indice de l'intervalle
    élémentaire [a/b^d, (a+1)/b^d) contenant x : a = floor(x * b^d).
    """
    return int(np.floor(x_coord * (b ** depth)))


# -----------------------------------------------------------------------------
# Approche Dick-Feischl avec digital nets
# -----------------------------------------------------------------------------

def compute_dick_feischl_weights(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    nu: int,
    m: int,
    b: int = 2,
    t_alpha: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les poids W_{X,P,nu,l} et W_{X,Y,P,nu,l} de la méthode
    Dick-Feischl avec digital net.

    Formules (3.1) et (3.2) de Göttlich et al. (2025) :
        W_{X,P,nu,l} = (b^{nu-m} / N) * sum_{q=0}^{min(s-1,nu)}
                          (-1)^q * C(s-1,q) * (1/b^q) * S_{nu-q}(z_l)
    où S_r(z_l) = sum_{d : |d| = r} |{n : x_n dans I_d(z_l)}|.

    Cette implémentation suit directement la définition par inclusion-exclusion
    (boucle sur les multi-indices), de coût O(b^m * m^s * N) ; elle est adaptée
    au régime s <= 5, m <= 10 utilisé dans l'étude.

    Parameters
    ----------
    X : np.ndarray, shape (N, s)
    Y : np.ndarray, shape (N,)
    P : np.ndarray, shape (L, s)
        Points du digital net (L = b^m).
    nu : int
        Paramètre d'averaging. Doit satisfaire m - t_alpha >= nu.
    m : int
        log_b(L).
    b : int
        Base du net.
    t_alpha : int
        Paramètre t du (t, m, s)-net.

    Returns
    -------
    W_X : np.ndarray, shape (L,)
    W_XY : np.ndarray, shape (L,)
    """
    N, s = X.shape
    L = P.shape[0]
    assert L == b ** m, f"L doit être b^m = {b ** m}, reçu {L}."
    assert m - t_alpha >= nu, f"Condition m - t_alpha >= nu violée."

    W_X = np.zeros(L)
    W_XY = np.zeros(L)

    # Pour chaque ordre r = nu - q, énumère tous les multi-indices d in N_0^s
    # avec |d| = r. Pour chaque d et chaque l, compte les points X_n in I_d(z_l).
    for q in range(min(s - 1, nu) + 1):
        r = nu - q
        sign = (-1) ** q * comb(s - 1, q) / (b ** q)

        # Énumération des d in N_0^s avec |d| = r
        for d_tuple in _multi_indices_with_sum(s, r):
            d_arr = np.array(d_tuple, dtype=np.int64)

            # Pour chaque l, identifier les n avec x_{n,j} et z_{l,j} dans
            # le même intervalle élémentaire en dimension j, pour tout j.
            # Astuce : on identifie chaque point par son tuple d'indices
            # (floor(x_j * b^{d_j}) pour chaque j), puis on groupe.

            def cell_id(points):
                # points shape (M, s) -> tuples d'indices
                idx = np.zeros(points.shape, dtype=np.int64)
                for j in range(s):
                    idx[:, j] = np.floor(points[:, j] * (b ** d_arr[j])).astype(np.int64)
                return idx

            X_cells = cell_id(X)  # (N, s)
            Z_cells = cell_id(P)  # (L, s)

            # Pour chaque z_l, S_r(z_l) compte les x_n avec X_cells[n] == Z_cells[l]
            # Approche efficace via dictionnaire
            from collections import defaultdict
            counter_X = defaultdict(int)
            sum_Y = defaultdict(float)
            for n in range(N):
                key = tuple(X_cells[n].tolist())
                counter_X[key] += 1
                sum_Y[key] += Y[n]

            for l in range(L):
                key = tuple(Z_cells[l].tolist())
                W_X[l] += sign * counter_X.get(key, 0)
                W_XY[l] += sign * sum_Y.get(key, 0.0)

    # Facteur global b^{nu - m} / N
    factor = (b ** (nu - m)) / N
    W_X *= factor
    W_XY *= factor
    return W_X, W_XY


def _multi_indices_with_sum(s: int, r: int):
    """
    Énumère récursivement tous les multi-indices (d_1, ..., d_s) in N_0^s
    avec d_1 + ... + d_s = r.
    """
    if s == 1:
        yield (r,)
        return
    for d1 in range(r + 1):
        for rest in _multi_indices_with_sum(s - 1, r - d1):
            yield (d1,) + rest


def approximate_loss_averaging(
    f: callable,
    Z: np.ndarray,
    W_X: np.ndarray,
    W_XY: np.ndarray,
    Y_sumsq_over_N: float,
) -> float:
    """
    Calcule appL(f) selon la formule (1.3) :
        appL(f) = sum_l f(z_l)^2 * W_X[l] - 2 * sum_l f(z_l) * W_XY[l]
                 + (1/N) * sum_n y_n^2

    Parameters
    ----------
    f : callable, prend (L, d) -> (L,)
    Z : np.ndarray, shape (L, d)
    W_X, W_XY : poids pré-calculés
    Y_sumsq_over_N : (1/N) * sum_n Y_n^2 (calculé une fois pour toutes)
    """
    fz = f(Z)
    return float(np.sum(fz ** 2 * W_X) - 2.0 * np.sum(fz * W_XY) + Y_sumsq_over_N)


if __name__ == "__main__":
    # Test rapide en 2D avec L=16 = 2^4
    rng = np.random.default_rng(0)
    N = 500
    s = 2
    m = 4
    b = 2
    L = b ** m
    nu = 2

    X = rng.uniform(0, 1, size=(N, s))
    Y = (X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2

    from src.digital_nets import sobol_points
    P = sobol_points(L=L, d=s)

    print(f"Calcul des poids Dick-Feischl pour N={N}, L={L}, s={s}, nu={nu}...")
    W_X, W_XY = compute_dick_feischl_weights(X, Y, P, nu=nu, m=m, b=b)
    print(f"  sum(W_X) = {W_X.sum():.4f}  (devrait être proche de 1)")
    print(f"  sum(W_XY) = {W_XY.sum():.4f}  (devrait être proche de mean(Y) = {Y.mean():.4f})")

    # Test de l'approximation de err(f) pour f(x) = x_1 + x_2
    def f(z):
        return z[:, 0] + z[:, 1]

    Y_sumsq = float(np.mean(Y ** 2))
    err_exact = float(np.mean((f(X) - Y) ** 2))
    err_approx = approximate_loss_averaging(f, P, W_X, W_XY, Y_sumsq)
    print(f"\n  err exact = {err_exact:.6f}")
    print(f"  err approché (averaging) = {err_approx:.6f}")
    print(f"  écart absolu = {abs(err_exact - err_approx):.6f}")
