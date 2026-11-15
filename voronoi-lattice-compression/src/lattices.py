"""
Construction de règles de réseau rank-1 (rank-1 lattice rules).

Référence principale :
    Dick, Kritzer, Pillichshammer (2022),
    "Lattice Rules: Numerical Integration, Approximation, and Discrepancy",
    Springer.

    Nuyens & Cools (2006), "Fast algorithms for component-by-component
    construction of rank-1 lattice rules in shift-invariant reproducing
    kernel Hilbert spaces", Math. Comp. 75.

Notation suivie de Gnewuch, Harsha, Wnuk (2024) :
    Z = {z_ell = (ell / L) * g mod 1 : ell = 0, ..., L-1}
    où g = (g_1, ..., g_d) est le vecteur générateur dans {1, ..., L-1}^d.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# -----------------------------------------------------------------------------
# Construction des points du lattice à partir d'un vecteur générateur
# -----------------------------------------------------------------------------

def lattice_points(generator: np.ndarray, L: int) -> np.ndarray:
    """
    Construit les L points du rank-1 lattice associé au vecteur générateur.

    Parameters
    ----------
    generator : np.ndarray, shape (d,)
        Vecteur générateur g, avec composantes dans {1, ..., L-1}.
    L : int
        Nombre de points. Idéalement premier (les bornes théoriques de
        Gnewuch et al. supposent L premier).

    Returns
    -------
    Z : np.ndarray, shape (L, d)
        Points du lattice dans [0, 1)^d.
    """
    g = np.asarray(generator, dtype=np.int64)
    ell = np.arange(L, dtype=np.int64)
    # produit extérieur ell * g, puis modulo L, puis division par L
    Z = (np.outer(ell, g) % L) / L
    return Z


# -----------------------------------------------------------------------------
# Worst-case error pour l'espace de Korobov pondéré (forme close)
# -----------------------------------------------------------------------------

def _bernoulli_polynomial(x: np.ndarray, order: int) -> np.ndarray:
    """
    Évalue le polynôme de Bernoulli B_{2*alpha} sur la grille x in [0,1).

    Pour les espaces de Korobov H_{alpha,gamma,d} avec alpha entier,
    on a phi_alpha(x) = -(-1)^alpha * (2*pi)^{2*alpha} / (2*alpha)! * B_{2*alpha}(x).
    Ici on retourne directement phi_alpha pour alpha = order (entier).

    Parameters
    ----------
    x : np.ndarray
        Points dans [0, 1).
    order : int
        Ordre alpha (entier >= 1). On évalue B_{2*alpha}.

    Returns
    -------
    np.ndarray
        Valeurs de phi_alpha(x).
    """
    # Formules pour alpha = 1, 2, 3, 4 (cas usuels en QMC)
    # phi_alpha(x) = sum_{h != 0} exp(2*pi*i*h*x) / |h|^{2*alpha}
    # Pour alpha entier, phi_alpha s'exprime via B_{2*alpha}(x).
    if order == 1:
        # B_2(x) = x^2 - x + 1/6
        # phi_1(x) = 2*pi^2 * B_2({x})
        B = x ** 2 - x + 1.0 / 6.0
        return 2.0 * np.pi ** 2 * B
    elif order == 2:
        # B_4(x) = x^4 - 2*x^3 + x^2 - 1/30
        # phi_2(x) = -((2*pi)^4 / 24) * B_4({x})
        B = x ** 4 - 2.0 * x ** 3 + x ** 2 - 1.0 / 30.0
        return - (2.0 * np.pi) ** 4 / 24.0 * B
    elif order == 3:
        # B_6(x) = x^6 - 3*x^5 + (5/2)*x^4 - (1/2)*x^2 + 1/42
        B = (x ** 6 - 3.0 * x ** 5 + 2.5 * x ** 4
             - 0.5 * x ** 2 + 1.0 / 42.0)
        return (2.0 * np.pi) ** 6 / 720.0 * B
    else:
        raise NotImplementedError(
            f"Polynôme de Bernoulli B_{{2*{order}}} non implémenté. "
            "Ajoutez la formule explicite si nécessaire."
        )


def worst_case_error_korobov(
    generator: np.ndarray,
    L: int,
    alpha: int,
    gamma: np.ndarray,
) -> float:
    """
    Calcule l'erreur worst-case dans l'espace de Korobov pondéré H_{alpha,gamma,d}
    pour le rank-1 lattice de générateur g et taille L.

    Formule close (voir Kuo 2003, Théorème 4) :
        e(H,Z)^2 = -1 + (1/L) * sum_{ell=0}^{L-1} prod_{j=1}^{d}
                                 (1 + gamma_j * phi_alpha(ell*g_j/L mod 1))

    Parameters
    ----------
    generator : np.ndarray, shape (d,)
    L : int
    alpha : int
        Paramètre de régularité (entier >= 1).
    gamma : np.ndarray, shape (d,)
        Poids produit gamma_1, ..., gamma_d.

    Returns
    -------
    float
        Erreur worst-case e(H, Z) (positive).
    """
    g = np.asarray(generator, dtype=np.int64)
    gamma = np.asarray(gamma, dtype=np.float64)
    d = len(g)
    ell = np.arange(L, dtype=np.float64)

    # Pour chaque coordonnée j, calcule phi_alpha((ell * g_j / L) mod 1)
    # Shape: (L, d)
    frac = (np.outer(ell, g) % L) / L
    phi = _bernoulli_polynomial(frac, alpha)

    # Produit sur les d composantes : (1 + gamma_j * phi_j)
    factors = 1.0 + gamma[np.newaxis, :] * phi
    prod = np.prod(factors, axis=1)

    val = -1.0 + np.mean(prod)
    # val devrait être >= 0 ; tronqué à 0 pour la robustesse numérique
    return float(np.sqrt(max(val, 0.0)))


# -----------------------------------------------------------------------------
# Algorithme Component-By-Component (CBC) classique
# -----------------------------------------------------------------------------

def cbc_construction(
    L: int,
    d: int,
    alpha: int,
    gamma: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Construit un vecteur générateur via l'algorithme CBC classique.

    Complexité : O(d * L^2).

    Parameters
    ----------
    L : int
        Nombre de points (idéalement premier).
    d : int
        Dimension.
    alpha : int
        Régularité (entier >= 1).
    gamma : np.ndarray, optionnel, shape (d,)
        Poids produit. Par défaut, gamma_j = 1 pour tout j.
    verbose : bool
        Affiche la progression si True.

    Returns
    -------
    g : np.ndarray, shape (d,)
        Vecteur générateur.
    """
    if gamma is None:
        gamma = np.ones(d)
    gamma = np.asarray(gamma, dtype=np.float64)

    g = np.zeros(d, dtype=np.int64)
    g[0] = 1  # convention standard

    # Candidats pour chaque composante : entiers premiers avec L dans {1,...,L-1}.
    # Si L est premier, c'est juste {1, ..., L-1}.
    candidates = np.array(
        [k for k in range(1, L) if np.gcd(k, L) == 1],
        dtype=np.int64,
    )

    for s in range(1, d):
        best_err = np.inf
        best_k = candidates[0]
        for k in candidates:
            g_trial = g.copy()
            g_trial[s] = k
            err = worst_case_error_korobov(
                g_trial[: s + 1], L, alpha, gamma[: s + 1]
            )
            if err < best_err:
                best_err = err
                best_k = k
        g[s] = best_k
        if verbose:
            print(f"  CBC : composante {s+1}/{d} = {best_k}, err = {best_err:.6e}")

    return g


if __name__ == "__main__":
    # Test rapide
    L = 101  # premier
    d = 3
    alpha = 2
    print(f"Construction CBC pour L={L}, d={d}, alpha={alpha}...")
    g = cbc_construction(L, d, alpha, verbose=True)
    print(f"Vecteur générateur : {g}")
    Z = lattice_points(g, L)
    print(f"Points du lattice : shape = {Z.shape}")
    print(f"Premiers points :\n{Z[:5]}")
    err = worst_case_error_korobov(g, L, alpha, np.ones(d))
    print(f"Erreur worst-case : {err:.6e}")
