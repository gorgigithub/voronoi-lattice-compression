"""
Méthode Gnewuch-Harsha-Wnuk (2024) : compression QMC supervisée via
rank-1 lattices et troncature de séries de Fourier.

Formules clés (cf. Section 2 de l'article) :

Les poids sont définis par
    phi_K(z_ell) = (1/N) * sum_{k in K} sum_{n=1}^{N} c_n * chi_k(x_n - z_ell)

où chi_k(x) = exp(2*pi*i * k . x), c_n = 1 pour W_X et c_n = y_n pour W_XY.

Pour K = Q^alpha_{m, gamma, d} (croix hyperbolique en escalier), Section 3.6.3,
on utilise l'astuce du noyau de Dirichlet (formule (38)) :

    phi_Q(z_ell) = (1/N) * sum_n c_n
                          * sum_{t in T(m,d)} D_{kappa_up_1}(x_{n,1} - z_{ell,1})
                              * prod_{j=2}^{d} [D_{kappa_up_j} - D_{kappa_low_j}]
                                              (x_{n,j} - z_{ell,j})

où D_K(x) = sum_{k=-K}^{K} e^{2*pi*i*k*x} = sin(pi*(2K+1)*x) / sin(pi*x)
est le noyau de Dirichlet 1D.

Coût : O(N * L * d * |T(m,d)|), où |T(m,d)| = C(m+d-1, d-1).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from src.index_sets import (
    step_hyperbolic_cross_decomposition,
    rectangle_indices,
    rectangle_size,
    kappa_bound,
)


# -----------------------------------------------------------------------------
# Noyau de Dirichlet 1D
# -----------------------------------------------------------------------------

def dirichlet_kernel(x: np.ndarray, K: int) -> np.ndarray:
    """
    D_K(x) = sum_{k=-K}^{K} exp(2*pi*i*k*x)
           = sin(pi*(2K+1)*x) / sin(pi*x)  (avec D_K(0) = 2K+1 par continuité).

    Convention : K >= 0 et x peut être quelconque (typiquement x in [-1, 1]).
    Pour K = -1, D_{-1} = 0 (convention adoptée dans Gnewuch et al. au
    voisinage de l'origine pour la première coordonnée des blocs).

    Parameters
    ----------
    x : np.ndarray
        Points d'évaluation.
    K : int
        Borne du noyau.

    Returns
    -------
    np.ndarray, valeurs réelles (car la somme est symétrique).
    """
    if K < 0:
        return np.zeros_like(x)
    if K == 0:
        return np.ones_like(x)

    # Évaluation numériquement stable : sin(pi*(2K+1)*x) / sin(pi*x)
    # avec gestion du cas x = 0 (mod 1).
    pix = np.pi * x
    sin_denom = np.sin(pix)
    # Cas |sin(pi*x)| très petit -> formule limite (2K+1)
    eps = 1e-14
    out = np.empty_like(x)
    safe = np.abs(sin_denom) > eps
    out[safe] = np.sin((2 * K + 1) * pix[safe]) / sin_denom[safe]
    out[~safe] = 2 * K + 1
    return out


# -----------------------------------------------------------------------------
# Poids GHW pour la croix hyperbolique en escalier (méthode principale)
# -----------------------------------------------------------------------------

def compute_ghw_weights_step_hc(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    m: int,
    gamma: np.ndarray,
    alpha: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les poids de Gnewuch-Harsha-Wnuk pour la croix hyperbolique
    en escalier Q^alpha_{m, gamma, d}, en utilisant l'astuce du noyau de
    Dirichlet (formule (38)).

    Parameters
    ----------
    X : np.ndarray, shape (N, d) -- données
    Y : np.ndarray, shape (N,)   -- réponses
    Z : np.ndarray, shape (L, d) -- points du rank-1 lattice
    m : int                      -- paramètre de niveau (taille du cross)
    gamma : np.ndarray, shape (d,) -- poids produit
    alpha : int                  -- régularité

    Returns
    -------
    W_X : np.ndarray, shape (L,)  -- poids pour la somme f^2
    W_XY : np.ndarray, shape (L,) -- poids pour la somme y*f

    Note
    ----
    Coût naïf : O(N * L * d * |T(m,d)|). Pour d et m raisonnables ceci
    reste tractable. L'article suggère plusieurs accélérations possibles
    (Section 3.6.3), mais cette implémentation directe permet déjà de
    valider la correction et de comparer expérimentalement.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    N, d = X.shape
    L = Z.shape[0]
    assert Z.shape[1] == d

    # Décomposition de Q^alpha_{m, gamma, d} en blocs rectangulaires dyadiques
    decomposition = step_hyperbolic_cross_decomposition(m, gamma, alpha)

    W_X = np.zeros(L, dtype=np.float64)
    W_XY = np.zeros(L, dtype=np.float64)

    # Pour chaque bloc t et chaque z_ell, on calcule
    #   (1/N) * sum_n c_n * prod_j [D_up_j - D_low_j](x_{n,j} - z_{ell,j})
    # avec la convention low_1 = -1 (donc D_{-1} = 0 -> facteur = D_up_1).
    #
    # L'astuce de vectorisation : pour un bloc t fixé, on calcule la matrice
    # F[n, ell, j] = facteur_j(x_{n,j} - z_{ell,j}) puis le produit sur j.
    # Cela évite la boucle sur n et ell. Coût mémoire O(N*L*d) par bloc.

    for t, intervals in decomposition:
        # Pour chaque dimension j, prépare le facteur 1D
        # factor_j[n, ell] = D_up(x_{n,j} - z_{ell,j}) si j == 0
        #                  = D_up - D_low                 sinon
        # On accumule le produit dans contrib[n, ell].
        contrib = np.ones((N, L), dtype=np.float64)
        for j in range(d):
            lo, up = intervals[j]
            # diff[n, ell] = x_{n,j} - z_{ell,j}
            diff = X[:, j:j + 1] - Z[:, j].reshape(1, L)
            D_up = dirichlet_kernel(diff, up)
            if j == 0:
                factor = D_up
            else:
                D_lo = dirichlet_kernel(diff, lo)
                factor = D_up - D_lo
            contrib *= factor

        # Accumulation dans W_X et W_XY
        # W_X[ell]  += (1/N) * sum_n 1   * contrib[n, ell]
        # W_XY[ell] += (1/N) * sum_n Y_n * contrib[n, ell]
        W_X += contrib.sum(axis=0) / N
        W_XY += (Y[:, np.newaxis] * contrib).sum(axis=0) / N

    return W_X, W_XY


# -----------------------------------------------------------------------------
# Poids GHW pour le rectangle (variante plus simple, Section 3.5)
# -----------------------------------------------------------------------------

def compute_ghw_weights_rectangle(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    nu: float,
    gamma: np.ndarray,
    alpha: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les poids GHW pour K = R^alpha_{nu, gamma, d} (rectangle pondéré),
    formule (33) de l'article :

        phi_R(z_ell) = (1/N) * sum_n c_n * prod_j D_{kappa*_j}(x_{n,j} - z_{ell,j})

    où kappa*_j = floor((gamma_j * nu)^{1/(2*alpha)}).

    Coût : O(N * L * d), indépendant de |R| -- c'est l'avantage principal
    du rectangle (Section 3.5.3).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    N, d = X.shape
    L = Z.shape[0]

    kappa = [kappa_bound(gamma[j], nu, alpha) for j in range(d)]

    contrib = np.ones((N, L), dtype=np.float64)
    for j in range(d):
        diff = X[:, j:j + 1] - Z[:, j].reshape(1, L)
        contrib *= dirichlet_kernel(diff, kappa[j])

    W_X = contrib.sum(axis=0) / N
    W_XY = (Y[:, np.newaxis] * contrib).sum(axis=0) / N
    return W_X, W_XY


# -----------------------------------------------------------------------------
# Variante NAÏVE pour vérification : sommation directe sur l'ensemble K
# -----------------------------------------------------------------------------

def compute_ghw_weights_naive_rectangle(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    nu: float,
    gamma: np.ndarray,
    alpha: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcul direct (lent) des poids pour le rectangle, en sommant explicitement
    sur tous les multi-indices k in R. Utile pour vérifier que
    compute_ghw_weights_rectangle (via noyau de Dirichlet) donne le même
    résultat.

    Coût : O(N * L * |R|), très lent.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    N, d = X.shape
    L = Z.shape[0]

    W_X = np.zeros(L, dtype=np.complex128)
    W_XY = np.zeros(L, dtype=np.complex128)

    for k in rectangle_indices(nu, gamma, alpha):
        k_arr = np.array(k, dtype=np.float64)
        # chi_k(x_n - z_ell) pour tous n, ell
        # = exp(2*pi*i * k . (x_n - z_ell))
        phase_X = 2 * np.pi * (X @ k_arr)      # (N,)
        phase_Z = 2 * np.pi * (Z @ k_arr)      # (L,)
        # chi[n, ell] = exp(i * (phase_X[n] - phase_Z[ell]))
        chi = np.exp(1j * (phase_X[:, None] - phase_Z[None, :]))
        W_X += chi.sum(axis=0) / N
        W_XY += (Y[:, None] * chi).sum(axis=0) / N

    # Les poids doivent être réels (symétrie k <-> -k)
    return W_X.real, W_XY.real


# -----------------------------------------------------------------------------
# Approximation de la perte
# -----------------------------------------------------------------------------

def approximate_loss_ghw(
    f,
    Z: np.ndarray,
    W_X: np.ndarray,
    W_XY: np.ndarray,
    Y_sumsq_over_N: float,
) -> float:
    """
    Formule (3) de Gnewuch-Harsha-Wnuk (2024) / (1.3) de Göttlich et al. :

        appL(f) = (1/L) sum_ell f(z_ell)^2 * W_X[ell]
                - 2*(1/L) sum_ell f(z_ell) * W_XY[ell]
                + (1/N) * sum_n y_n^2

    Le facteur 1/L vient de la règle de quadrature des lattices.
    """
    L = Z.shape[0]
    fz = f(Z)
    T1 = np.sum(fz ** 2 * W_X) / L
    T2 = np.sum(fz * W_XY) / L
    return float(T1 - 2.0 * T2 + Y_sumsq_over_N)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Test 1 : noyau de Dirichlet
    print("=== Test du noyau de Dirichlet ===")
    x = np.array([0.0, 0.1, 0.25, 0.5])
    for K in [0, 1, 2, 3]:
        D = dirichlet_kernel(x, K)
        # vérification : D_K(0) = 2K+1
        print(f"  K={K} : D_K(x) = {D}, D_K(0) attendu = {2*K+1}")

    # Test 2 : équivalence noyau de Dirichlet vs sommation naïve (rectangle)
    print("\n=== Test équivalence Dirichlet vs naïf (rectangle) ===")
    N, d = 200, 2
    L = 8
    X = rng.uniform(0, 1, size=(N, d))
    Y = rng.standard_normal(N)
    Z = rng.uniform(0, 1, size=(L, d))
    gamma = np.ones(d)
    alpha = 1
    nu = 4.0

    W_X_fast, W_XY_fast = compute_ghw_weights_rectangle(X, Y, Z, nu, gamma, alpha)
    W_X_naive, W_XY_naive = compute_ghw_weights_naive_rectangle(
        X, Y, Z, nu, gamma, alpha
    )

    err_X = np.max(np.abs(W_X_fast - W_X_naive))
    err_XY = np.max(np.abs(W_XY_fast - W_XY_naive))
    print(f"  Erreur max W_X (Dirichlet vs naïf) : {err_X:.2e}")
    print(f"  Erreur max W_XY (Dirichlet vs naïf) : {err_XY:.2e}")
    print(f"  -> {'OK' if max(err_X, err_XY) < 1e-10 else 'PROBLÈME'}")

    # Test 3 : approximation de la perte pour f(x) = sin(2*pi*x1)
    print("\n=== Test approximation de la perte (rectangle + lattice) ===")
    from src.lattices import cbc_construction, lattice_points
    L = 31  # premier
    g = cbc_construction(L, d, alpha=2)
    Z = lattice_points(g, L)

    def f(x):
        return np.sin(2 * np.pi * x[:, 0])

    err_exact = float(np.mean((f(X) - Y) ** 2))
    Y_sumsq = float(np.mean(Y ** 2))

    for nu in [4.0, 16.0, 64.0]:
        W_X, W_XY = compute_ghw_weights_rectangle(X, Y, Z, nu, gamma, alpha=2)
        err_app = approximate_loss_ghw(f, Z, W_X, W_XY, Y_sumsq)
        size = rectangle_size(nu, gamma, alpha=2)
        print(f"  nu={nu:5.1f} (|R|={size}) : "
              f"err exact = {err_exact:.4f}, "
              f"err approché = {err_app:.4f}, "
              f"écart = {abs(err_exact - err_app):.4e}")

    # Test 4 : step hyperbolic cross
    print("\n=== Test step HC ===")
    for m in [2, 3, 4]:
        W_X, W_XY = compute_ghw_weights_step_hc(X, Y, Z, m, gamma, alpha=2)
        err_app = approximate_loss_ghw(f, Z, W_X, W_XY, Y_sumsq)
        print(f"  m={m} : err exact = {err_exact:.4f}, "
              f"err approché = {err_app:.4f}, "
              f"écart = {abs(err_exact - err_app):.4e}")
