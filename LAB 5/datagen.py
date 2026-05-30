import numpy as np


def generate_blobs(means, covs, sizes, seed=42):
    """Sample points from several 2-D Gaussians N(mean_i, cov_i).

    Parameters
    ----------
    means : list of (2,) centres
    covs  : list of (2, 2) covariance matrices
    sizes : list of ints, points per component
    seed  : RNG seed (reproducibility)

    Returns
    -------
    X : (sum(sizes), 2) points, shuffled
    y : (sum(sizes),)   true component index of each point
    """
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for i, (m, c, n) in enumerate(zip(means, covs, sizes)):
        Xs.append(rng.multivariate_normal(np.asarray(m, float), np.asarray(c, float), size=n))
        ys.append(np.full(n, i))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(X))          # shuffle so ordering carries no signal
    return X[perm], y[perm]
