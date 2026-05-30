# K-Means Clustering & Gaussian Mixture Model (NumPy from scratch)

Implementation of **K-Means** and a **Gaussian Mixture Model (GMM)** using **NumPy only**
for the algorithms (matplotlib only for plotting). Both are trained with the
**Expectation–Maximization (EM)** method, and every result is evaluated **quantitatively**.

## Design note — structure follows the problem

There is **no dataset file to load**: the toy data is *generated* from the Gaussian
specifications in the assignment. So the modules are named and split by what this problem
actually does — two algorithms, a data generator, metrics, and plots — rather than a
generic data-pipeline skeleton (no `data_loader`, `preprocessing`, or `config.yaml`,
which would not fit synthetic-data tasks).

```
kmeans-gmm/
├── README.md
├── requirements.txt
├── .gitignore
│
├── kmeans.py                # KMeans class      (NumPy, EM/Lloyd)
├── gmm.py                   # GaussianMixture   (NumPy, EM, full covariance, random init)
├── datagen.py               # generate_blobs (toy data)
├── metrics.py               # accuracy (best permutation), ARI, purity, distortion
├── plots.py                 # cluster / ellipse / convergence plots
│
├── kmeans_assignment1.py    # 3 equal clusters + effect of random init
├── kmeans_assignment2.py    # unequal cluster sizes (1200 / 200 / 1000)
├── kmeans_assignment3.py    # anisotropic cluster (Σ = diag(10, 1))
├── gmm_assignment1.py       # implement & train the GMM with EM
├── run_all.py
└── figures/                 # PNG outputs
```

Each `*_assignment*.py` defines its dataset **inline** (the spec is quoted in the
docstring, so script ↔ requirement is obvious), trains the model, prints metrics +
comments, and saves figures. The library modules have no side effects on import.

## Setup & run

Python 3.9+. Run from inside this folder.

```bash
pip install -r requirements.txt

python kmeans_assignment1.py
python kmeans_assignment2.py
python kmeans_assignment3.py
python gmm_assignment1.py

python run_all.py                    # all four
```

## Quantitative results (seed 42)

Cluster ids are arbitrary, so the supervised scores in `metrics.py` are
permutation-invariant: **accuracy** (best label match over all K! permutations),
**Adjusted Rand Index** (chance-corrected, 1 = perfect), **purity**, and the K-means
**distortion J**.

| | accuracy | ARI |
|---|---|---|
| A1 — equal clusters | 0.98 | 0.95 |
| A2 — unequal sizes | 0.98 | 0.93 |
| A3 — anisotropic, **K-means** | 0.93 | 0.81 |
| A3 — anisotropic, **GMM** | 0.98 | 0.94 |

The A3 row is the headline: K-means degrades on the elongated cluster, and the GMM with
full covariances recovers it.

## Implementation notes

- **K-Means** minimises `J = Σ_n Σ_k r_nk ||x_n − μ_k||²`; E-step assigns each point to
  its nearest centroid, M-step sets each centroid to the mean of its members. Supports
  random / custom init and re-seeds empty clusters.
- **GMM** uses **full covariances** and the EM updates for `γ_k, μ_k, Σ_k, π_k`. The
  E-step runs in log-space (log-sum-exp) for stability; a small `reg_covar` keeps
  covariances invertible. Parameters are **initialised randomly** exactly as the course
  material specifies (random data points as means, global covariance, uniform weights);
  EM is repeated `n_init` times and the highest-likelihood run is kept. No dependence on
  any other model.

## Library compliance

The two algorithm files (`kmeans.py`, `gmm.py`) import **NumPy only**. The whole project
uses nothing beyond NumPy + matplotlib (plotting) + Python's standard library. No
`sklearn`, `scipy`, `pandas`, or any external model.
