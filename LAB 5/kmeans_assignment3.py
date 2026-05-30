"""
kmeans_assignment3.py
=====================
K-Means Assignment 3.
- Generate 200 ~ N((2,2), S1), 200 ~ N((8,3), S1), 200 ~ N((3,6), S2)
  with S1 = [[1,0],[0,1]] and S2 = [[10,0],[0,1]]  (third blob elongated).
- Implement K-means (NumPy), train with EM.
- Comment on the effect of the anisotropic distribution N((3,6), S2).

Run:  python kmeans_assignment3.py
"""

import numpy as np

from datagen import generate_blobs
from kmeans import KMeans
from metrics import clustering_report
from plots import figpath, save_pair

MEANS = [(2, 2), (8, 3), (3, 6)]
S1 = [[1, 0], [0, 1]]
S2 = [[10, 0], [0, 1]]             # stretched along x
SIZES = [200, 200, 200]
SEED = 42


def main():
    X, y = generate_blobs(MEANS, [S1, S1, S2], SIZES, seed=SEED)

    km = min((KMeans(3, random_state=s).fit(X) for s in range(20)), key=lambda m: m.inertia_)

    rep = clustering_report(y, km.labels_, X, km.cluster_centers_)
    print(f"accuracy = {rep['accuracy']:.3f} | ARI = {rep['ARI']:.3f} | "
          f"purity = {rep['purity']:.3f}")

    big = (y == 2)
    pred_big = km.labels_[big]
    dominant = np.bincount(pred_big).argmax()
    print(f"elongated cluster: {int((pred_big != dominant).sum())} / {int(big.sum())} "
          f"points split off to other clusters")

    save_pair(X, (y, np.array(MEANS, float), "Ground truth (3rd blob stretched)"),
              (km.labels_, km.cluster_centers_, "K-means result"),
              figpath("as3_result.png"))
    print("\nfigure -> as3_result.png")
    print(COMMENTS)


COMMENTS = """
COMMENTS - effect of the elongated distribution N((3,6), S2)
-------------------------------------------------------------
S2 = diag(10, 1) stretches the third blob to +/- sqrt(10) ~ +/-3.2 along x, so it spans
x in [0, 6] at y ~ 6 and physically overlaps the (2,2) blob and reaches toward (8,3).
K-means uses plain Euclidean distance and therefore assumes spherical, equal-variance
clusters; it can only draw straight equidistant boundaries. The long tails end up closer
to NEIGHBOURING centroids than to their own, so they are re-assigned: the elongated
cluster is cut along its long axis and its ends absorbed by the other two clusters (see
the count above and the lower accuracy/ARI). Takeaway: K-means fails on anisotropic
clusters; a GMM with full covariances can model the shape and assign the tails correctly
(see gmm_assignment1.py).
"""


if __name__ == "__main__":
    main()
