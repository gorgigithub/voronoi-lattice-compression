"""
Fonctions test pour l'analyse d'erreur d'approximation.

Deux familles :

(A) Fonctions de Göttlich et al. (2025) :
  - f1 : produit-exponentielle (continue, V(f1) < inf, mais pas C^alpha)
  - f2 : exponentielle tronquée (discontinue, V(f2) < inf)
  - f3 : fonction de Zhou (mélange gaussien, très régulière)

(B) Famille de Genz (1984), standard en quadrature QMC. Six fonctions,
    chacune testant un "défi" différent. Toutes définies sur [0,1]^d avec
    un vecteur de paramètres a (difficulté) et u (décalage) :
  - g_oscillatory   : oscillations
  - g_product_peak  : pic localisé (produit)
  - g_corner_peak   : pic dans un coin
  - g_gaussian      : pic gaussien
  - g_continuous    : continue non lisse (type C^0)
  - g_discontinuous : discontinue

Références :
  A. Genz (1984), "Testing multidimensional integration routines",
  Tools, Methods and Languages for Scientific and Engineering Computation.
  Surjanovic & Bingham, Virtual Library of Simulation Experiments,
  https://www.sfu.ca/~ssurjano/integration.html
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Famille (A) : Göttlich et al. (2025)
# =============================================================================

def f1(x: np.ndarray, a: np.ndarray = None, u: np.ndarray = None) -> np.ndarray:
    """f1(x) = exp(-sum_i a_i * |x_i - u_i|). Continue, non différentiable.

    NOTE : même forme fonctionnelle que la fonction « continuous » de Genz
    (alias g5 / genz_continuous). La seule différence est le choix des
    paramètres : ici a_i = 5 (Göttlich et al.), contre a_i ≈ 2.04/d pour g5.
    Tester f1 ET g5 revient donc à tester cette forme à deux niveaux de
    difficulté (test de robustesse au paramètre)."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    if a is None:
        a = 5.0 * np.ones(d)
    if u is None:
        u = 0.5 * np.ones(d)
    return np.exp(-np.sum(a * np.abs(x - u), axis=1))


def f2(x: np.ndarray, a: np.ndarray = None, u: np.ndarray = None) -> np.ndarray:
    """f2(x) = 0 si x_1>u_1 ou x_2>u_2, sinon exp(sum a_i x_i). Discontinue.

    NOTE : même forme fonctionnelle que la fonction « discontinuous » de Genz
    (alias g6 / genz_discontinuous). Seuls les paramètres diffèrent :
    ici a_i = 5 (Göttlich et al.), contre a_i ≈ 4.30/d pour g6. Tester f2 ET
    g6 revient à tester cette forme à deux niveaux de difficulté."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    if a is None:
        a = 5.0 * np.ones(d)
    if u is None:
        u = 0.5 * np.ones(d)
    mask = (x[:, 0] <= u[0]) & (x[:, 1] <= u[1])
    out = np.zeros(x.shape[0])
    out[mask] = np.exp(np.sum(a * x[mask], axis=1))
    return out


def f3_zhou(x: np.ndarray) -> np.ndarray:
    """Fonction de Zhou : mélange de 2 gaussiennes. Très régulière, localisée."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    coef = 10.0 ** d / 2.0
    phi = lambda y: (1.0 / (2.0 * np.pi) ** (d / 2)) * np.exp(-0.5 * np.sum(y ** 2, axis=1))
    return coef * (phi(10 * x - 10.0 / 3.0) + phi(10 * x - 20.0 / 3.0))


# =============================================================================
# Famille (B) : Genz (1984)
# =============================================================================
# Convention : a contrôle la "difficulté" (plus a est grand, plus c'est dur),
# u contrôle la localisation. Par défaut a_i normalisé pour que la difficulté
# soit comparable entre dimensions.

def _genz_params(d, a, u, a_default):
    if a is None:
        a = a_default * np.ones(d) / d  # normalisation par d (usuel chez Genz)
    else:
        a = np.atleast_1d(a)
        if a.size == 1:
            a = a.item() * np.ones(d)
    if u is None:
        u = 0.5 * np.ones(d)
    else:
        u = np.atleast_1d(u)
        if u.size == 1:
            u = u.item() * np.ones(d)
    return a, u


def g_oscillatory(x, a=None, u=None):
    """Genz 1 — oscillatoire : cos(2*pi*u_1 + sum a_i x_i)."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    a, u = _genz_params(d, a, u, a_default=9.0)
    return np.cos(2 * np.pi * u[0] + x @ a)


def g_product_peak(x, a=None, u=None):
    """Genz 2 — pic produit : prod_i 1/(a_i^-2 + (x_i - u_i)^2)."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    a, u = _genz_params(d, a, u, a_default=7.25)
    return np.prod(1.0 / (a ** (-2) + (x - u) ** 2), axis=1)


def g_corner_peak(x, a=None, u=None):
    """Genz 3 — pic de coin : (1 + sum a_i x_i)^{-(d+1)}."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    a, u = _genz_params(d, a, u, a_default=1.85)
    return (1.0 + x @ a) ** (-(d + 1))


def g_gaussian(x, a=None, u=None):
    """Genz 4 — pic gaussien : exp(-sum a_i^2 (x_i - u_i)^2)."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    a, u = _genz_params(d, a, u, a_default=7.03)
    return np.exp(-np.sum(a ** 2 * (x - u) ** 2, axis=1))


def g_continuous(x, a=None, u=None):
    """Genz 5 — continue non lisse : exp(-sum a_i |x_i - u_i|)."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    a, u = _genz_params(d, a, u, a_default=2.04)
    return np.exp(-np.sum(a * np.abs(x - u), axis=1))


def g_discontinuous(x, a=None, u=None):
    """Genz 6 — discontinue : 0 si x_1>u_1 ou x_2>u_2, sinon exp(sum a_i x_i)."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    a, u = _genz_params(d, a, u, a_default=4.30)
    out = np.exp(x @ a)
    if d >= 2:
        mask = (x[:, 0] > u[0]) | (x[:, 1] > u[1])
    else:
        mask = (x[:, 0] > u[0])
    out[mask] = 0.0
    return out


# =============================================================================
# Registre
# =============================================================================

_TABLE = {
    # Göttlich et al.
    "f1": f1,
    "f2": f2,
    "f3": f3_zhou,
    # Genz
    "genz_oscillatory": g_oscillatory,
    "genz_product_peak": g_product_peak,
    "genz_corner_peak": g_corner_peak,
    "genz_gaussian": g_gaussian,
    "genz_continuous": g_continuous,
    "genz_discontinuous": g_discontinuous,
}

# Alias courts
_ALIASES = {
    "g1": "genz_oscillatory",
    "g2": "genz_product_peak",
    "g3": "genz_corner_peak",
    "g4": "genz_gaussian",
    "g5": "genz_continuous",
    "g6": "genz_discontinuous",
}


def get_test_function(name: str):
    """Retourne la fonction par nom (ou alias)."""
    if name in _ALIASES:
        name = _ALIASES[name]
    if name not in _TABLE:
        raise ValueError(
            f"Fonction inconnue : {name}. "
            f"Choix : {list(_TABLE)} ou alias {list(_ALIASES)}."
        )
    return _TABLE[name]


def list_test_functions():
    """Liste tous les noms de fonctions disponibles."""
    return list(_TABLE.keys())


# Caractérisation de la régularité (pour l'analyse)
REGULARITY = {
    "f1": "continue, non lisse (C^0)",
    "f2": "discontinue",
    "f3": "très lisse (C^inf), localisée",
    "genz_oscillatory": "lisse (C^inf), oscillante",
    "genz_product_peak": "lisse, pic localisé",
    "genz_corner_peak": "lisse, pic de coin",
    "genz_gaussian": "lisse (C^inf), pic gaussien",
    "genz_continuous": "continue, non lisse (C^0)",
    "genz_discontinuous": "discontinue",
}


# =============================================================================
# Formes fonctionnelles distinctes
# =============================================================================
# Parmi les 9 fonctions, certaines partagent la MÊME forme analytique avec des
# paramètres différents. Il n'y a en réalité que 7 formes distinctes. Ce
# dictionnaire regroupe les fonctions par forme : chaque clé est une forme,
# chaque valeur est la liste des noms qui l'implémentent (variantes de
# paramètres). Utile pour une présentation honnête dans un article.

DISTINCT_FORMS = {
    "exp_continue": {
        "formule": "exp(-sum a_i |x_i - u_i|)",
        "regularite": "continue, non lisse (C^0)",
        "variantes": ["f1", "genz_continuous"],   # f1: a=5 ; g5: a≈2.04/d
    },
    "exp_tronquee": {
        "formule": "exp(sum a_i x_i) tronquée",
        "regularite": "discontinue",
        "variantes": ["f2", "genz_discontinuous"],  # f2: a=5 ; g6: a≈4.30/d
    },
    "melange_gaussien": {
        "formule": "(10^d/2)(phi(10x-10/3) + phi(10x-20/3))",
        "regularite": "très lisse (C^inf), localisée",
        "variantes": ["f3"],
    },
    "oscillatoire": {
        "formule": "cos(2pi u_1 + sum a_i x_i)",
        "regularite": "lisse (C^inf), oscillante",
        "variantes": ["genz_oscillatory"],
    },
    "pic_produit": {
        "formule": "prod_i 1/(a_i^-2 + (x_i - u_i)^2)",
        "regularite": "lisse, pic localisé",
        "variantes": ["genz_product_peak"],
    },
    "pic_coin": {
        "formule": "(1 + sum a_i x_i)^-(d+1)",
        "regularite": "lisse, pic de coin",
        "variantes": ["genz_corner_peak"],
    },
    "pic_gaussien": {
        "formule": "exp(-sum a_i^2 (x_i - u_i)^2)",
        "regularite": "lisse (C^inf), pic gaussien",
        "variantes": ["genz_gaussian"],
    },
}


def list_distinct_forms():
    """
    Retourne une fonction représentative par forme distincte (7 au total).
    Pour l'exponentielle continue et l'exponentielle tronquée, on prend la
    variante de Göttlich (f1, f2) par défaut.
    """
    return [info["variantes"][0] for info in DISTINCT_FORMS.values()]


def get_form_of(name: str) -> str:
    """Retourne le nom de la forme à laquelle appartient une fonction."""
    if name in _ALIASES:
        name = _ALIASES[name]
    for form, info in DISTINCT_FORMS.items():
        if name in info["variantes"]:
            return form
    return "inconnue"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for d in [2, 3]:
        X = rng.uniform(0, 1, size=(4, d))
        print(f"\n=== Dimension {d} ===")
        for name in list_test_functions():
            f = get_test_function(name)
            vals = f(X)
            print(f"  {name:22s} {REGULARITY[name]:32s} -> "
                  f"min={vals.min():.3f}, max={vals.max():.3f}")
