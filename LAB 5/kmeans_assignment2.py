"""
kmeans_assignment2.py
=====================
K-Means Assignment 2.
- Generate 1200 ~ N((2,2), I), 200 ~ N((8,3), I), 1000 ~ N((3,6), I).
- Implement K-means (NumPy), train with EM.
- Comment on the effect of different cluster sizes.

Run:  python kmeans_assignment2.py
"""

import numpy as np

from datagen import generate_blobs
from kmeans import KMeans
from metrics import clustering_report
from plots import figpath, save_pair

MEANS = [(2, 2), (8, 3), (3, 6)]
COV = [[1, 0], [0, 1]]
SIZES = [1200, 200, 1000]          # imbalanced
SEED = 42


def main():
    X, y = generate_blobs(MEANS, [COV] * 3, SIZES, seed=SEED)
    print("true cluster sizes:", SIZES, " total =", len(X))

    km = min((KMeans(3, random_state=s).fit(X) for s in range(20)), key=lambda m: m.inertia_)
    print("predicted cluster sizes:", np.bincount(km.labels_).tolist())

    rep = clustering_report(y, km.labels_, X, km.cluster_centers_)
    print(f"accuracy = {rep['accuracy']:.3f} | ARI = {rep['ARI']:.3f} | "
          f"purity = {rep['purity']:.3f} | J = {rep['distortion']:.1f}")

    save_pair(X, (y, np.array(MEANS, float), "Ground truth (1200/200/1000)"),
              (km.labels_, km.cluster_centers_, "K-means result"),
              figpath("as2_result.png"))
    print("\nfigure -> as2_result.png")
    print(COMMENTS)


COMMENTS = """
COMMENTS - effect of unequal cluster sizes
-------------------------------------------
K-means minimises the TOTAL sum of squared distances, so large clusters dominate the
objective and pull boundaries toward themselves, capturing border points that belong to
a smaller neighbour. Here the (2,2) [1200] and (3,6) [1000] blobs are large and only
moderately separated, so their overlap is split by a straight bisector rather than by
true membership (a modest drop in accuracy/ARI vs Assignment 1). The small (8,3) cluster
[200] is well isolated and survives; near a large blob it could be swallowed, or a big
blob split in two, because that lowers J more. K-means implicitly assumes roughly equal
sizes and equal spherical variance, with no prior weight pi_k - exactly what a GMM adds.
"""


if __name__ == "__main__":
    main()
