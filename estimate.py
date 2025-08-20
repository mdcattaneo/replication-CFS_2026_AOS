import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, eigh
from typing import List, Callable, Sequence, Tuple, Iterable, Dict


class Link:
    def __init__(self, link_type: str):
        self.link_type: str = link_type

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        if self.link_type == "logistic":
            return 1.0 / (1.0 + np.exp(-t))
        if self.link_type == "log-log":
            return 1 - np.exp(-np.exp(t))
        if self.link_type == "id":
            return t

    def derivative(self, t: float | np.ndarray) -> float | np.ndarray:
        if self.link_type == "logistic":
            s = self(t)
            return s * (1.0 - s)
        if self.link_type == "log-log":
            return np.exp(t) * np.exp(-np.exp(t))
        elif self.link_type == "id":
            return np.ones(t.shape)
        raise ValueError(f"link_type = {self.link_type} is not supported")

    def inverse(self, s: float | np.ndarray) -> float | np.ndarray:
        if self.link_type == "logistic":
            return np.log(s / (1.0 - s))
        if self.link_type == "log-log":
            return np.log(-np.log(1 - s))
        elif self.link_type == "id":
            return s
        raise ValueError(f"link_type = {self.link_type} is not supported")


def d_link_of_mu(F_cond, link):
    """Return vector η'(η^{-1}(F(q|x))) evaluated at all rows of X."""
    def _d(q, X):
        s = F_cond(q, X)
        s = np.clip(s, 1e-12, 1-1e-12)           # guard inverse
        mu = link.inverse(s)                      # μ = η^{-1}(F)
        return link.derivative(mu)                # η'(μ)
    return _d


def d_link_of_mu_squared(F_cond: Callable[[float, np.ndarray], float], link: Link) -> Callable[[float, np.ndarray], float]:
    link_type = link.link_type
    if link_type == "logistic":
        return lambda q, x: (F_cond(q, x) * (1.0 - F_cond(q, x)))**2
    if link_type == "log-log":
        return lambda q, x: ((1 - F_cond(q, x)) * np.log(1 - F_cond(q, x)))**2
    elif link_type == "id":
        return lambda q, x: np.ones(x.shape[0])
    raise ValueError(f"link_type = {link_type} is not supported")


def optimise_beta(
    y: np.ndarray,              # shape (n,)
    x: np.ndarray,              # shape (n, d)
    P_X: np.ndarray,
    q: float,
    link: Link,
    beta0: np.ndarray | None = None,
    method: str = "L-BFGS-B",
    **scipy_kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Non-linear least squares estimator for distribution regression.

    Parameters
    ----------
    y, x        : data vectors/matrix (n observations)
    basis       : list of tensor-product spline basis functions
    q           : query point (scalar)
    eta, d_eta  : link and derivative; default = logistic
    beta0       : initial guess (defaults to zeros)
    method      : any scipy.optimize.minimize method that accepts jac
    scipy_kwargs: optional arguments forwarded to `minimize`

    Returns
    -------
    beta_hat    : argmin estimate (1-D ndarray, length K)
    info        : scipy optimisation result dict
    """
    y = np.asarray(y).ravel()
    x = np.asarray(x)
    n, K = P_X.shape

    y_ind = (y <= q).astype(float)          # indicator vector

    # --- objective and gradient --------------------------------------------
    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        z = P_X @ beta                 # shape (n,)
        mu = link(z)                  # eta(z)
        residuals = y_ind - mu
        loss = np.sum(residuals**2)

        # gradient: -2 Σ r_i * eta'(z_i) * p_i
        grad = -2.0 * (link.derivative(z) * residuals) @ P_X   # shape (K,)
        return loss, grad

    if beta0 is None:
        beta0 = np.zeros(K)

    res = minimize(
        fun=lambda b: objective(b)[0],
        x0=beta0,
        jac=lambda b: objective(b)[1],
        method=method,
        **scipy_kwargs,
    )

    return res.x, res.__dict__


def _flatten_beta(beta_hat) -> np.ndarray:
    """
    Turn beta_hat into a 1-D float array, even if it is a nested list/tuple
    of arrays with different lengths.
    """
    # common case: already 1-D ndarray
    if isinstance(beta_hat, np.ndarray) and beta_hat.ndim == 1:
        return beta_hat.astype(float, copy=False)

    # if it's a (list|tuple) of ndarrays / lists, concat along the first axis
    if isinstance(beta_hat, (list, tuple)):
        try:
            return np.concatenate([np.asarray(b, dtype=float).ravel()
                                   for b in beta_hat])
        except ValueError:
            raise ValueError("Could not flatten beta_hat; make sure it is "
                             "a 1-D array or a list/tuple of same-rank arrays.")

    # fallback
    return np.asarray(beta_hat, dtype=float).ravel()


def make_muhat(
    basis: Sequence[Callable[[np.ndarray], float]],
    beta_hat,
) -> Callable[[np.ndarray | Iterable[np.ndarray]], np.ndarray]:
    """
    Build μ̂(x) = Σ_k β̂_k p_k(x).

    Returns a vectorised callable:
        • Input  shape (d,)  → scalar
        • Input  shape (n,d) → (n,) vector
    """
    beta = _flatten_beta(beta_hat)
    K = len(basis)
    if beta.shape[0] != K:
        raise ValueError(f"beta_hat length {beta.shape[0]} "
                         f"does not match number of basis functions {K}.")

    def mu_hat(x: np.ndarray | Iterable[np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        single_point = (x.ndim == 1)
        if single_point:
            x = x[None, :]                      # promote to shape (1, d)

        n = x.shape[0]
        P = np.empty((n, K), dtype=float)       # design matrix
        for k, f in enumerate(basis):
            # evaluate kth basis function on all rows of x
            P[:, k] = [f(row) for row in x]

        yhat = P @ beta                         # shape (n,)
        return yhat[0] if single_point else yhat

    return mu_hat


# ------------------------------------------------------------
def make_design_matrix(X: np.ndarray,
                        basis: Sequence[Callable[[np.ndarray], float]]
                        ) -> np.ndarray:
    """Return P[i,k] = p_k(X[i]).  Shape: (n, K)."""
    if not X.ndim == 2:
        raise ValueError(f"X must be a 2D array of shape (n, d), got {X.shape}")
    n, K = X.shape[0], len(basis)
    P = np.empty((n, K), dtype=float)
    for k, f in enumerate(basis):
        P[:, k] = [f(xi) for xi in X]          # evaluate kth basis on all rows
    return P


def compute_gram(
    P_X: np.ndarray,
):
    n, K = P_X.shape
    return P_X.T @ P_X / n


# ---------- Qbar(q) computation ------------------------------
def compute_qbar(
    X: np.ndarray,
    q: float,
    P_X: np.ndarray,
    F_cond: Callable[[float, np.ndarray], float],
    link: Link,
) -> np.ndarray:
    n, K = P_X.shape

    d_eta_sq = d_link_of_mu_squared(F_cond, link)(q, X)  # shape (n,)

    Qbar = np.einsum('ni,n,nj->ij', P_X, 2 * d_eta_sq / n, P_X)
    return Qbar


# ---------- Q̂(q) computation ------------------------------
def compute_qhat(
    P_X: np.ndarray,
    beta_hat: np.ndarray,
    link: Link,
) -> np.ndarray:
    """
    Compute  Q̂(q) = 2/n Σ η'(μ̂(x_i;q))² p(x_i)p(x_i)ᵀ.

    Returns a (K,K) NumPy array.
    """
    n, K = P_X.shape
    z = P_X @ beta_hat                           # μ̂(x_i;q), shape (n,)
    w = link.derivative(z)**2                            # weights η'(·)², shape (n,)

    # Efficient: multiply each row of P by w_i, then accumulate
    Qhat = (2.0 / n) * P_X.T @ (w[:, None] * P_X)   # (K, K)
    return Qhat


def compute_sigmabar(
    X: np.ndarray,
    q: float,
    P_X: np.ndarray,
    F_cond: Callable[[float, np.ndarray], float],
    link: Link,
) -> np.ndarray:
    n, K = P_X.shape
    d_eta_sq = d_link_of_mu_squared(F_cond, link)(q, X)  # shape (n,)
    weights = 4.0 * F_cond(q, X) * (1.0 - F_cond(q, X)) * d_eta_sq  # shape (n,)
    Sigmabar = np.einsum('ni,n,nj->ij', P_X, weights / n, P_X)
    return Sigmabar


# ---------- Σ̂(q) computation ------------------------------
def compute_sigmahat(
    P_X: np.ndarray,
    beta_hat: np.ndarray,                       # length K
    link: Link,
) -> np.ndarray:
    """
    Σ̂(q) = (4/n) Σ  η(μ̂)·(1-η(μ̂))·η'(μ̂) · p(x)p(x)ᵀ
           where μ̂ = p(x)ᵀ β̂.
    Returns a (K, K) matrix.
    """
    n, K = P_X.shape
    muhat  = P_X @ beta_hat                          # shape (n,)
    weights = 4.0 * link(muhat) * (1.0 - link(muhat)) * link.derivative(muhat)**2   # shape (n,)
    Sigmahat = np.einsum('ni,n,nj->ij', P_X, weights / n, P_X)
    return Sigmahat


# ------------ Ω̂(q) evaluator ----------------------------------------------
def make_omegahat(
    basis: Sequence[Callable[[np.ndarray], float]],
    Qhat_inv: np.ndarray,            # shape (K, K)
    Sigmahat: np.ndarray,        # shape (K, K)
):
    """
    Returns a callable Ω̂_q(x). which for x of shape (n,d) returns an (n,) vector.
    """
    M = Qhat_inv @ Sigmahat @ Qhat_inv       # pre-compute K × K matrix

    def omegahat(x: np.ndarray | Iterable[np.ndarray]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not x.ndim == 2:
            raise ValueError(f"x must be a 2D array of shape (n, d), got {x.shape}")
        P_x = make_design_matrix(x, basis)

        # element-wise quadratic form: row_k · M · row_kᵀ
        vals = np.einsum("ij,jk,ik->i", P_x, M, P_x)
        return vals

    return omegahat


def compute_sigmabar_q1_q2(
        q1: float, q2: float,
        F_cond: Callable[[float, np.ndarray], float],
        X: np.ndarray,
        P_X: np.ndarray,
        link: Link,
    ) -> np.ndarray:
    """Return K×K matrix Σ̄(q1,q2)."""
    # element-wise weights for every observation
    if q1 > q2:
        q1, q2 = q2, q1

    n = P_X.shape[0]

    eta_min = F_cond(q1, X)
    eta_maxc = 1.0 - F_cond(q2, X)
    S_i = 4.0 * eta_min * eta_maxc
    w_i = S_i * d_link_of_mu(F_cond, link)(q1, X) * d_link_of_mu(F_cond, link)(q2, X)
    return np.einsum('ni,n,nj->ij', P_X, w_i / n, P_X)


def compute_sigmahat_q1_q2(
        q1: float, q2: float,
        mu_hat_dict: Dict[float, np.ndarray],
        d_eta_hat_dict: Dict[float, np.ndarray],
        P_X: np.ndarray,
        link: Link,
    ) -> np.ndarray:
    """Return K×K matrix Σ̂(q1,q2)."""
    # element-wise weights for every observation
    n = P_X.shape[0]
    if q1 > q2:
        q1, q2 = q2, q1
    m1, m2 = mu_hat_dict[q1], mu_hat_dict[q2]
    eta_min  = link(m1)
    eta_maxc = 1.0 - link(m2)
    S_i   = 4.0 * eta_min * eta_maxc               # Ŝ_i
    w_i   = S_i * d_eta_hat_dict[q1] * d_eta_hat_dict[q2]      # total weight
    # Σ̂ = (1/n) Σ w_i p_i p_iᵀ
    return (P_X.T * w_i) @ P_X / n


def build_sigmahat_block_matrix(
    q_grid: Sequence[float],                       # length N_q
    K: int,
    Sigmahat_map: Callable[[float, float], np.ndarray],
) -> np.ndarray:
    """
    Return (K N_q) × (K N_q) matrix with blocks
        Σ_hat(q_i, q_j)  in lexicographic order of q_grid.
    """
    Nq = len(q_grid)
    Sigma_big = np.empty((K * Nq, K * Nq), dtype=float)

    for j2, q2 in enumerate(q_grid):
        for j1, q1 in enumerate(q_grid):
            # (K,K) block
            blk = Sigmahat_map(float(q1), float(q2))

            r = slice(j1 * K, (j1 + 1) * K)        # row indices
            c = slice(j2 * K, (j2 + 1) * K)        # col indices

            Sigma_big[r, c] = blk

    return Sigma_big


def construct_z_fields(
    Zhat_K: np.ndarray,                     # K·N_q × num_of_exps_for_Z
    C_q_sqrtOm_dict: Dict[float, np.ndarray],  # q → (N_x, K)
    grid_q: Sequence[float],
) -> np.ndarray:
    """
    Returns Z_field of length N_x·N_q in lexicographic order
    """
    N_x, K = C_q_sqrtOm_dict[grid_q[0]].shape
    N_q = len(grid_q)
    num_of_exps_for_Z = Zhat_K.shape[1]
    Z_field = np.empty((N_x * N_q, num_of_exps_for_Z))
    for j, q in enumerate(grid_q):
        Z_q = Zhat_K[j * K : (j + 1) * K]
        proj = np.einsum('lk,ke->le', C_q_sqrtOm_dict[q], Z_q)
        assert proj.shape == (N_x, num_of_exps_for_Z)
        Z_field[j * N_x : (j + 1) * N_x, :] = proj

    return Z_field  # shape (N_x * N_q, num_of_exps_for_Z)