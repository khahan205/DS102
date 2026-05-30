import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


def figpath(name):
    d = os.path.join(_HERE, "figures")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def plot_clusters(X, labels, centers, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=12, alpha=0.6)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X",
                   s=220, edgecolor="black", linewidths=1.4, label="centroid")
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_aspect("equal", "box")
    return ax


def save_pair(X, left, right, path):
    """left/right = (labels, centers, title). Save a 2-panel comparison figure."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, (lbl, ctr, ttl) in zip(ax, (left, right)):
        plot_clusters(X, lbl, ctr, ttl, a)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_gmm_ellipses(ax, gmm, n_std=2.0):
    """Draw the n_std-sigma ellipse + mean of every GMM component."""
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(t), np.sin(t)])
    for k in range(gmm.K):
        vals, vecs = np.linalg.eigh(gmm.Sigma_[k])
        ell = (vecs @ (n_std * np.sqrt(vals)[:, None] * circle)) + gmm.mu_[k][:, None]
        ax.plot(ell[0], ell[1], "r-", lw=2)
        ax.scatter(*gmm.mu_[k], c="red", marker="X", s=150, edgecolor="k", zorder=5)


def plot_convergence(values, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, "o-")
    ax.set_xlabel("EM iteration"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
