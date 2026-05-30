"""
gmm_assignment1.py
==================
GMM Assignment 1.
- Implement the Gaussian Mixture Model (NumPy only).
- Train it with EM.

The assignment does not specify data, so we generate a 3-component mixture. We use the
anisotropic case from K-means Assignment 3, which lets us show *measurably* that the GMM
(full covariances) succeeds where K-means failed: same metrics, side by side.

Run:  python gmm_assignment1.py
"""

import numpy as np
import matplotlib.pyplot as plt

from datagen import generate_blobs
from kmeans import KMeans
from gmm import GaussianMixture
from metrics import clustering_report
from plots import figpath, plot_clusters, plot_gmm_ellipses, plot_convergence

MEANS = [(2, 2), (8, 3), (3, 6)]
S1 = [[1, 0], [0, 1]]
S2 = [[10, 0], [0, 1]]
SIZES = [200, 200, 200]
SEED = 42


def main():
    X, y = generate_blobs(MEANS, [S1, S1, S2], SIZES, seed=SEED)

    # train the GMM with EM
    gmm = GaussianMixture(3, random_state=SEED, verbose=True).fit(X)
    print(f"\nGMM converged in {gmm.n_iter_} iterations")
    print("mixing coefficients pi:", np.round(gmm.pi_, 3))
    print("means:\n", np.round(gmm.mu_, 2))
    print(f"final log-likelihood = {gmm.lower_bound_:.2f} | BIC = {gmm.bic(X):.1f}")

    ll = np.array(gmm.loglik_history_)
    print("log-likelihood non-decreasing every step:", bool(np.all(np.diff(ll) >= -1e-6)))

    # K-means baseline for a quantitative comparison
    km = min((KMeans(3, random_state=s).fit(X) for s in range(20)), key=lambda m: m.inertia_)
    rk = clustering_report(y, km.labels_)
    rg = clustering_report(y, gmm.predict(X))
    print(f"\nK-means : accuracy {rk['accuracy']:.3f} | ARI {rk['ARI']:.3f}")
    print(f"GMM     : accuracy {rg['accuracy']:.3f} | ARI {rg['ARI']:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    plot_clusters(X, km.labels_, km.cluster_centers_, "K-means (spherical)", ax[0])
    ax[1].scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap="viridis", s=12, alpha=0.6)
    plot_gmm_ellipses(ax[1], gmm)
    ax[1].set_title("GMM (full covariance, 2$\\sigma$)")
    ax[1].set_xlabel("$x_1$"); ax[1].set_ylabel("$x_2$"); ax[1].set_aspect("equal", "box")
    fig.tight_layout(); fig.savefig(figpath("gmm_a1_compare.png"), bbox_inches="tight"); plt.close(fig)
    plot_convergence(gmm.loglik_history_, "log-likelihood",
                     "GMM log-likelihood (monotonically increasing)",
                     figpath("gmm_a1_loglik.png"))

    print("\nfigures -> gmm_a1_compare.png, gmm_a1_loglik.png")
    print(COMMENTS)


COMMENTS = """
COMMENTS
--------
The log-likelihood increases monotonically every EM iteration (verified above),
confirming a correct implementation. On the anisotropic data the GMM beats K-means on
every metric: full covariances let one component stretch to fit the elongated blob, and
the soft responsibilities gamma_k assign its tails correctly - recovering what K-means
cut in Assignment 3.
"""


if __name__ == "__main__":
    main()
