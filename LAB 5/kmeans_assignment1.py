"""
kmeans_assignment1.py
=====================
K-Means Assignment 1.
- Generate 600 points: 200 ~ N((2,2), I), 200 ~ N((8,3), I), 200 ~ N((3,6), I).
- Implement K-means (NumPy), train with EM.
- Comment on the effect of random centroid initialisation.

Run:  python kmeans_assignment1.py
"""

import numpy as np
import matplotlib.pyplot as plt

from datagen import generate_blobs
from kmeans import KMeans
from metrics import clustering_report
from plots import figpath, plot_clusters, save_pair, plot_convergence

# --- dataset specified by the assignment ---
MEANS = [(2, 2), (8, 3), (3, 6)]
COV = [[1, 0], [0, 1]]
SIZES = [200, 200, 200]
SEED = 42
N_RESTARTS = 20


def main():
    X, y = generate_blobs(MEANS, [COV] * 3, SIZES, seed=SEED)
    print("dataset shape:", X.shape)

    # train K-means with EM (best of several random restarts)
    models = [KMeans(3, random_state=s).fit(X) for s in range(N_RESTARTS)]
    inertias = np.array([m.inertia_ for m in models])
    km = models[int(inertias.argmin())]
    print(f"best model: {km.n_iter_} iters, distortion J = {km.inertia_:.2f}")

    rep = clustering_report(y, km.labels_, X, km.cluster_centers_)
    print(f"accuracy = {rep['accuracy']:.3f} | ARI = {rep['ARI']:.3f} | "
          f"purity = {rep['purity']:.3f}")

    save_pair(X, (y, np.array(MEANS, float), "Ground truth"),
              (km.labels_, km.cluster_centers_, "K-means result"),
              figpath("as1_result.png"))

    # --- effect of random initialisation ---
    print(f"\nover {N_RESTARTS} restarts:  best J = {inertias.min():.2f}, "
          f"worst J = {inertias.max():.2f}")
    print(f"{int((inertias > inertias.min()*1.05).sum())} / {N_RESTARTS} restarts "
          f"reached a clearly worse local optimum")

    worst = models[int(inertias.argmax())]
    bad = KMeans(3, init=X[y == 0][:3]).fit(X)        # all 3 seeds inside one true blob

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_clusters(X, km.labels_, km.cluster_centers_, f"Best (J={km.inertia_:.0f})", ax[0])
    plot_clusters(X, worst.labels_, worst.cluster_centers_, f"Worst (J={worst.inertia_:.0f})", ax[1])
    plot_clusters(X, bad.labels_, bad.cluster_centers_, f"Bad init (J={bad.inertia_:.0f})", ax[2])
    fig.tight_layout(); fig.savefig(figpath("as1_random_init.png"), bbox_inches="tight"); plt.close(fig)
    plot_convergence(bad.history_, "distortion $J$", "Distortion of a bad-init run",
                     figpath("as1_convergence.png"))

    print("\nfigures -> as1_result.png, as1_random_init.png, as1_convergence.png")
    print(COMMENTS)


COMMENTS = """
COMMENTS - effect of random centroid initialisation
----------------------------------------------------
K-means EM only decreases J, so it always converges, but only to a LOCAL minimum that
depends on the starting centroids. These equal, well-separated, isotropic blobs are
easy, so most random restarts recover the true clusters (accuracy/ARI near 1). But a
bad draw - two seeds inside the same true blob - splits that blob and merges two real
clusters, giving a higher J and wrong labels (right panel). Fix: run several restarts
and keep the lowest-J result (done above), or use k-means++ seeding.
"""


if __name__ == "__main__":
    main()
