"""
gmm.py
------
Gaussian Mixture Model with full covariances, NumPy only, trained with EM
(exactly the algorithm in the `preliminaries` notebook):

    1. Initialise mu_k, Sigma_k, pi_k RANDOMLY, then evaluate the log-likelihood.
    2. E-step : gamma_k(x_n) = pi_k N(x_n|mu_k,Sigma_k) / sum_m pi_m N(x_n|mu_m,Sigma_m)
    3. M-step : N_k = sum_n gamma_k(x_n)
                mu_k    = (1/N_k) sum_n gamma_k(x_n) x_n
                Sigma_k = (1/N_k) sum_n gamma_k(x_n) (x_n-mu_k)(x_n-mu_k)^T
                pi_k    = N_k / N
    4. Evaluate the log-likelihood; go to step 2 until it stops increasing.

Initialisation is RANDOM, as the course material specifies (no dependence on any
other model). Because random starts can land in poor local optima, EM is repeated
`n_init` times from different random seeds and the run with the highest final
log-likelihood is kept - a standard, fully self-contained strategy.

Implemented with NumPy only. BIC / AIC are provided for model selection.
"""

import numpy as np


def _logsumexp(a, axis):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis)


class GaussianMixture:
    """Gaussian Mixture Model with full covariances, trained by EM (random init)."""

    def __init__(self, n_components=3, max_iter=200, tol=1e-4, reg_covar=1e-6,
                 n_init=10, random_state=None, verbose=False):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------- initialise
    def _initialise(self, X, rng):
        """RANDOM initialisation (per the preliminaries):
        - means  : K distinct random data points
        - covs   : the global data covariance (a stable scale), same for all k
        - weights: uniform 1/K
        """
        N, D = X.shape
        idx = rng.choice(N, self.K, replace=False)
        self.mu_ = X[idx].copy()
        global_cov = np.atleast_2d(np.cov(X.T)) + self.reg_covar * np.eye(D)
        self.Sigma_ = np.repeat(global_cov[None, :, :], self.K, axis=0)
        self.pi_ = np.full(self.K, 1.0 / self.K)

    # ----------------------------------------------- log N(x|mu_k,Sigma_k)
    def _log_gaussian(self, X):
        N, D = X.shape
        log_pdf = np.empty((N, self.K))
        for k in range(self.K):
            diff = X - self.mu_[k]
            cov = self.Sigma_[k] + self.reg_covar * np.eye(D)
            _, logdet = np.linalg.slogdet(cov)
            inv = np.linalg.inv(cov)
            maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
            log_pdf[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + maha)
        return log_pdf

    # ------------------------------------------------------------- EM steps
    def _e_step(self, X):
        log_w = np.log(self.pi_ + 1e-300) + self._log_gaussian(X)
        log_norm = _logsumexp(log_w, axis=1)
        log_resp = log_w - log_norm[:, None]
        return np.exp(log_resp), log_norm.sum()

    def _m_step(self, X, resp):
        N, D = X.shape
        Nk = resp.sum(axis=0) + 1e-300
        self.pi_ = Nk / N
        self.mu_ = (resp.T @ X) / Nk[:, None]
        for k in range(self.K):
            diff = X - self.mu_[k]
            self.Sigma_[k] = (resp[:, k, None] * diff).T @ diff / Nk[k]
            self.Sigma_[k] += self.reg_covar * np.eye(D)

    def _run_em(self, X):
        """Run EM from the current initialisation; return the log-likelihood history."""
        history = []
        prev = -np.inf
        for it in range(self.max_iter):
            resp, loglik = self._e_step(X)        # E-step (+ log-likelihood)
            history.append(loglik)
            self._m_step(X, resp)                 # M-step
            if abs(loglik - prev) < self.tol:
                break
            prev = loglik
        self._last_n_iter = it + 1
        return history

    # ------------------------------------------------------------- training
    def fit(self, X):
        X = np.asarray(X, float)
        self._n_features_ = X.shape[1]
        best = None
        for run in range(self.n_init):
            seed = None if self.random_state is None else self.random_state + run
            rng = np.random.default_rng(seed)
            self._initialise(X, rng)
            history = self._run_em(X)
            final_ll = history[-1]
            if self.verbose:
                print(f"  init {run + 1}/{self.n_init}: final log-likelihood = {final_ll:.3f}")
            if best is None or final_ll > best["ll"]:
                best = {"ll": final_ll, "mu": self.mu_.copy(), "Sigma": self.Sigma_.copy(),
                        "pi": self.pi_.copy(), "history": history, "n_iter": self._last_n_iter}
        # restore the best run
        self.mu_, self.Sigma_, self.pi_ = best["mu"], best["Sigma"], best["pi"]
        self.loglik_history_ = best["history"]
        self.lower_bound_ = best["ll"]
        self.n_iter_ = best["n_iter"]
        return self

    # ------------------------------------------------------------- inference
    def predict_proba(self, X):
        resp, _ = self._e_step(np.asarray(X, float))
        return resp

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X):
        """Total log-likelihood of X under the fitted model."""
        _, loglik = self._e_step(np.asarray(X, float))
        return float(loglik)

    # ------------------------------------------------- model selection
    def _n_parameters(self):
        D = self._n_features_
        cov_params = self.K * D * (D + 1) / 2     # full symmetric covariances
        mean_params = self.K * D
        return int(cov_params + mean_params + self.K - 1)   # -1: weights sum to 1

    def bic(self, X):
        """Bayesian Information Criterion (lower is better)."""
        X = np.asarray(X, float)
        N = X.shape[0]
        return -2 * self.score(X) + self._n_parameters() * np.log(N)

    def aic(self, X):
        """Akaike Information Criterion (lower is better)."""
        return -2 * self.score(X) + 2 * self._n_parameters()
