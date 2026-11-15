"""
Méthodes de compression de données basées sur le clustering.

Implémente :
  - QMC-Voronoi (Göttlich, Heieck, Neuenkirch 2025) : centres = points QMC
    (digital net ou lattice), clusters = cellules de Voronoi, réponses
    compressées = moyennes des y_n dans chaque cellule.
  - supercompress / robust supercompress (Joseph & Mak 2021).

L'intérêt principal de ce module pour votre projet est la classe
VoronoiQMCCompressor, qui accepte n'importe quel ensemble de points
QMC (lattice OU digital net) et fait le clustering Voronoi correspondant.
C'est la brique de base de votre nouvelle méthode (Voronoi + lattices).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, Tuple


# -----------------------------------------------------------------------------
# Classe de base : interface commune pour toutes les méthodes de compression
# -----------------------------------------------------------------------------

class BaseCompressor:
    """
    Interface commune pour toutes les méthodes de compression supervisée.

    Méthode fit(X, Y) -> (Z, W) où :
        Z : points compressés, shape (L, d)
        W : réponses (ou poids) compressés, shape (L,) ou tuple de tableaux
    """

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def approximate_loss(
        self,
        f: callable,
        Z: np.ndarray,
        W: np.ndarray,
    ) -> float:
        """
        Calcule la perte approximée appL(f) en utilisant le jeu compressé.
        Pour les méthodes de type clustering :
            appL(f) = (1/L) * sum_l (f(z_l) - w_l)^2
        """
        f_vals = f(Z)
        return float(np.mean((f_vals - W) ** 2))


# -----------------------------------------------------------------------------
# QMC-Voronoi : LE compresseur central pour votre projet
# -----------------------------------------------------------------------------

class VoronoiQMCCompressor(BaseCompressor):
    """
    Compression QMC-Voronoi : les centres des clusters sont des points QMC
    pré-générés (digital net OU rank-1 lattice), et chaque cellule de Voronoi
    reçoit comme réponse la moyenne des y_n des points x_n qui y tombent.

    C'est la généralisation de l'algorithme (V) de Göttlich et al. (2025)
    où on autorise n'importe quel ensemble de centres QMC.

    Parameters
    ----------
    centers : np.ndarray, shape (L, d)
        Points QMC pré-générés (par lattice_points ou sobol_points, etc.).
    default_response : float
        Valeur attribuée à un cluster vide (par défaut 0.0).
    """

    def __init__(self, centers: np.ndarray, default_response: float = 0.0):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.L, self.d = self.centers.shape
        self.default_response = default_response

        # kd-tree pour assignations rapides
        self._tree = cKDTree(self.centers)

        # Remplis après fit
        self.assignments_: Optional[np.ndarray] = None
        self.cluster_sizes_: Optional[np.ndarray] = None
        self.responses_: Optional[np.ndarray] = None
        self.empty_clusters_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assigne chaque point x_n à son centre le plus proche, puis calcule
        la moyenne des y_n par cluster.

        Returns
        -------
        Z : np.ndarray, shape (L, d)
        W : np.ndarray, shape (L,)
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        N = X.shape[0]

        # Assignation au plus proche voisin parmi les centres
        _, assignments = self._tree.query(X, k=1)
        self.assignments_ = assignments

        # Moyenne des y_n par cluster (sum / size, en gérant les clusters vides)
        sums = np.zeros(self.L, dtype=np.float64)
        sizes = np.zeros(self.L, dtype=np.int64)
        np.add.at(sums, assignments, Y)
        np.add.at(sizes, assignments, 1)

        responses = np.full(self.L, self.default_response, dtype=np.float64)
        nonempty = sizes > 0
        responses[nonempty] = sums[nonempty] / sizes[nonempty]

        self.cluster_sizes_ = sizes
        self.responses_ = responses
        self.empty_clusters_ = np.where(~nonempty)[0]

        return self.centers, responses

    def empty_ratio(self) -> float:
        """
        Proportion de clusters vides (utile pour diagnostiquer si les centres
        QMC représentent bien la distribution des données).
        """
        if self.cluster_sizes_ is None:
            raise RuntimeError("Appeler fit() d'abord.")
        return float(np.mean(self.cluster_sizes_ == 0))


# -----------------------------------------------------------------------------
# supercompress (Joseph & Mak 2021) : clustering supervisé adaptatif
# -----------------------------------------------------------------------------

class SupercompressCompressor(BaseCompressor):
    """
    Algorithme supercompress de Joseph & Mak (2021).

    Construction itérative : on part de 2 clusters (k-means classique sur X),
    puis à chaque étape on subdivise le cluster qui a la plus grande perte
    sum_{i in I_k} (y_i - w_k)^2.

    Le paramètre lam permet d'interpoler entre supercompress (lam=0,
    purement supervisé) et k-means classique (lam=1, purement non supervisé).
    Joseph & Mak suggèrent lam = 1/(d+1) pour rsupercompress (robuste).

    Parameters
    ----------
    L : int
        Nombre de clusters cibles.
    lam : float in [0, 1]
        Compromis x-espace / y-espace. 0 = supercompress pur, 1 = k-means.
    random_state : int
        Pour la reproductibilité des k-means internes.
    """

    def __init__(
        self,
        L: int,
        lam: float = 0.0,
        random_state: int = 0,
    ):
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam doit être dans [0, 1].")
        self.L = L
        self.lam = lam
        self.random_state = random_state
        self.centers_: Optional[np.ndarray] = None
        self.responses_: Optional[np.ndarray] = None

    def _kmeans_split(
        self,
        X_sub: np.ndarray,
        Y_sub: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Subdivise un cluster en 2 via k-means.

        Suit la stratégie 'split 1' de l'Algorithme 1 de Joseph & Mak :
        clustering sur l'espace augmenté [X, Y*sqrt((1-lam)/lam)] si lam>0,
        ou pur k-means sur X si lam=1. Pour lam=0 (supercompress pur), on
        cluster sur Y uniquement.
        """
        from sklearn.cluster import KMeans

        if self.lam == 0.0:
            # supercompress : split sur Y (1D). On calcule directement le
            # meilleur seuil (variance intra-classe minimale) par tri, au lieu
            # d'un k-means 1D coûteux. La réassignation spatiale globale (par
            # kd-tree dans fit) garantit ensuite la cohérence des clusters.
            n = len(Y_sub)
            if n < 2:
                return (X_sub.mean(axis=0, keepdims=True),
                        np.array([Y_sub.mean()]),
                        np.zeros(n, dtype=int))
            order = np.argsort(Y_sub)
            ys = Y_sub[order]
            csum = np.cumsum(ys)
            csum2 = np.cumsum(ys ** 2)
            total = csum[-1]; total2 = csum2[-1]
            i = np.arange(1, n)
            left_sum = csum[:-1]; left_n = i
            right_sum = total - left_sum; right_n = n - i
            left_ss = csum2[:-1] - left_sum ** 2 / left_n
            right_ss = (total2 - csum2[:-1]) - right_sum ** 2 / right_n
            wcss = left_ss + right_ss
            cut = i[np.argmin(wcss)]
            labels = np.zeros(n, dtype=int)
            labels[order[cut:]] = 1
        elif self.lam == 1.0:
            # k-means classique : split sur X
            km = KMeans(n_clusters=2, n_init=4,
                        random_state=self.random_state)
            labels = km.fit_predict(X_sub)
        else:
            # robust : split sur [X, scaled Y]
            scale = np.sqrt((1 - self.lam) / self.lam)
            features = np.hstack([X_sub, scale * Y_sub.reshape(-1, 1)])
            km = KMeans(n_clusters=2, n_init=4,
                        random_state=self.random_state)
            labels = km.fit_predict(features)

        # Centres dans l'espace X (toujours), réponses = moyennes des y
        mask0 = labels == 0
        mask1 = labels == 1
        # Sécurité : si l'un des deux est vide, on annule le split
        if not mask0.any() or not mask1.any():
            # fallback : split arbitraire 50/50
            n = len(Y_sub)
            mask0 = np.zeros(n, dtype=bool)
            mask0[: n // 2] = True
            mask1 = ~mask0

        c0 = X_sub[mask0].mean(axis=0)
        c1 = X_sub[mask1].mean(axis=0)
        w0 = Y_sub[mask0].mean()
        w1 = Y_sub[mask1].mean()
        return np.vstack([c0, c1]), np.array([w0, w1]), labels

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        N, d = X.shape

        # Initialisation : 2 clusters
        centers, responses, _ = self._kmeans_split(X, Y)
        # Calcul des assignations et pertes initiales
        tree = cKDTree(centers)
        _, assignments = tree.query(X, k=1)
        # Pertes par cluster
        losses = np.zeros(2)
        for k in range(2):
            mask = assignments == k
            if mask.any():
                losses[k] = np.sum((Y[mask] - responses[k]) ** 2)

        # Itérations : subdiviser le pire cluster jusqu'à atteindre L clusters
        while len(centers) < self.L:
            k_worst = int(np.argmax(losses))
            mask = assignments == k_worst
            if mask.sum() < 2:
                # cluster non subdivisible : on l'écarte
                losses[k_worst] = -np.inf
                if np.all(losses == -np.inf):
                    break
                continue

            X_sub = X[mask]
            Y_sub = Y[mask]
            new_centers, new_responses, _ = self._kmeans_split(X_sub, Y_sub)

            # Remplacer le cluster k_worst par les 2 nouveaux
            centers = np.delete(centers, k_worst, axis=0)
            responses = np.delete(responses, k_worst)
            centers = np.vstack([centers, new_centers])
            responses = np.concatenate([responses, new_responses])

            # Re-assignation globale (kd-tree, vectorisé) puis mise à jour
            # des pertes SANS boucle Python (via np.add.at = bincount pondéré).
            tree = cKDTree(centers)
            _, assignments = tree.query(X, k=1)
            K = len(centers)
            # somme des Y et comptage par cluster, vectorisés
            sum_y = np.zeros(K)
            cnt = np.zeros(K)
            np.add.at(sum_y, assignments, Y)
            np.add.at(cnt, assignments, 1)
            mean_y = np.divide(sum_y, cnt, out=np.zeros(K), where=cnt > 0)
            # perte par cluster : sum(y^2) - cnt*mean^2  (variance non normalisée)
            sum_y2 = np.zeros(K)
            np.add.at(sum_y2, assignments, Y ** 2)
            losses = sum_y2 - cnt * mean_y ** 2
            losses[cnt == 0] = -np.inf

        self.centers_ = centers
        self.responses_ = responses
        return centers, responses




class SupercompressFaithful(BaseCompressor):
    """
    Implémentation FIDÈLE de l'Algorithme (S) de Göttlich et al. (2025, p.12),
    reproduisant la structure exacte de leur code Matlab public
    (fichier supercompress.m du dépôt associé à Göttlich et al. 2025).

    Structure du code Matlab reproduite à l'identique :
    - Initialisation : k-means en 2 clusters sur l'INPUT SPACE X.
    - Boucle : à chaque étape, on choisit le cluster de plus grande perte
      L_j = sum_{i in I_j}(y_i - w_j)^2, et on le subdivise par un k-means
      sur l'input space, UNIQUEMENT sur les points de ce cluster (split local,
      pas de réassignation globale).
    - Gestion du cluster à un seul point (on passe au suivant).

    Le K-means binaire est codé directement (Lloyd vectorisé) plutôt que via
    sklearn.KMeans, ce qui évite l'overhead fixe (~1ms/appel) des milliers
    d'appels et rend la comparaison Python-vs-Python plus représentative.

    Parameters
    ----------
    L : int, nombre de clusters cible.
    lam : float, compromis input/output (0 = supercompress, 1 = k-means).
    random_state : int, graine.
    """

    def __init__(self, L, lam=0.0, random_state=0):
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam doit être dans [0, 1].")
        self.L = L
        self.lam = lam
        self.random_state = random_state
        self.centers_ = None
        self.responses_ = None

    @staticmethod
    def _kmeans2(P, n_iter=10, seed=0):
        """K-means binaire maison sur P (n, d) -> labels {0, 1}."""
        n = len(P)
        if n < 2:
            return np.zeros(n, dtype=np.intp)
        rng = np.random.default_rng(seed)
        i0 = rng.integers(n)
        d2 = np.sum((P - P[i0]) ** 2, axis=1)
        if d2.max() == 0:
            return np.zeros(n, dtype=np.intp)
        i1 = int(np.argmax(d2))
        c = np.vstack([P[i0], P[i1]])
        labels = np.zeros(n, dtype=np.intp)
        for it in range(n_iter):
            d0 = np.sum((P - c[0]) ** 2, axis=1)
            d1 = np.sum((P - c[1]) ** 2, axis=1)
            new_labels = (d1 < d0).astype(np.intp)
            if it > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            m0 = labels == 0
            if m0.any():
                c[0] = P[m0].mean(axis=0)
            if (~m0).any():
                c[1] = P[~m0].mean(axis=0)
        return labels

    def _features(self, X, Y, idx):
        if self.lam == 0.0 or self.lam == 1.0:
            return X[idx]
        scale = np.sqrt((1 - self.lam) / self.lam)
        return np.hstack([X[idx], scale * Y[idx].reshape(-1, 1)])

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        N, d = X.shape

        init = self._kmeans2(self._features(X, Y, np.arange(N)),
                             seed=self.random_state)
        clusters = [np.where(init == 0)[0], np.where(init == 1)[0]]

        def loss(idx):
            if len(idx) == 0:
                return -np.inf
            return float(np.sum((Y[idx] - Y[idx].mean()) ** 2))

        losses = [loss(c) for c in clusters]
        seed = self.random_state
        while len(clusters) < self.L:
            k = int(np.argmax(losses))
            idx = clusters[k]
            if len(idx) < 2 or losses[k] <= 0:
                losses[k] = -np.inf
                if all(l == -np.inf for l in losses):
                    break
                continue
            seed += 1
            sub = self._kmeans2(self._features(X, Y, idx), seed=seed)
            idx0 = idx[sub == 0]; idx1 = idx[sub == 1]
            if len(idx0) == 0 or len(idx1) == 0:
                losses[k] = -np.inf
                if all(l == -np.inf for l in losses):
                    break
                continue
            clusters[k] = idx0; losses[k] = loss(idx0)
            clusters.append(idx1); losses.append(loss(idx1))

        self.centers_ = np.array([X[c].mean(axis=0) for c in clusters])
        self.responses_ = np.array([Y[c].mean() for c in clusters])
        return self.centers_, self.responses_



if __name__ == "__main__":
    # Petit test : fonction sin(2*pi*x) en 1D avec bruit
    rng = np.random.default_rng(42)
    N = 2000
    d = 1
    X = rng.uniform(0, 1, size=(N, d))
    Y = np.sin(2 * np.pi * X[:, 0]) + 0.05 * rng.standard_normal(N)

    # Centres QMC (juste une grille régulière ici, pour test)
    from src.lattices import lattice_points, cbc_construction
    g = cbc_construction(L=31, d=d, alpha=2)
    centers = lattice_points(g, L=31)

    print("Test VoronoiQMCCompressor :")
    comp = VoronoiQMCCompressor(centers)
    Z, W = comp.fit(X, Y)
    print(f"  Z.shape = {Z.shape}, W.shape = {W.shape}")
    print(f"  Proportion de clusters vides : {comp.empty_ratio():.2%}")
    print(f"  Première réponse moyennée : {W[0]:.4f}")

    print("\nTest SupercompressCompressor :")
    comp2 = SupercompressCompressor(L=31, lam=0.0)
    Z2, W2 = comp2.fit(X, Y)
    print(f"  Z2.shape = {Z2.shape}, W2.shape = {W2.shape}")
