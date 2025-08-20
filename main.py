import itertools
import logging
import json
import os
import time
from functools import wraps
from typing import Sequence, List, Tuple, Callable

import numpy as np
from scipy.stats import norm
from scipy.interpolate import BSpline
from scipy.linalg import cholesky

from estimate import (
    compute_gram, compute_qbar, compute_qhat,
    compute_sigmahat, compute_sigmahat_q1_q2, compute_sigmabar,
    compute_sigmabar_q1_q2, build_sigmahat_block_matrix,
    make_omegahat, make_design_matrix,
    optimise_beta, construct_z_fields,
    Link,
)

from sweep_definitions import get_args


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        logging.info(f'{f.__name__} starting')
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logging.debug(f'{f.__name__} finished: took {te - ts:2.4f} s')
        return result
    return wrap


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def construct_uniform_knots(p, s, J):
    res = [0.] * (p + 1)
    for i in range(1, J):
        res.extend((p - s + 1) * [i / J])
    res.extend([1. + 1e-8] * (p + 1))
    return res


def calc_K_s(p, s, J):
    return (p + 1) * J - s * (J - 1)


def construct_spline_basis(p, s, J, knots=None):
    # s - 1 times continuously diff, of order m = p + 1
    # for simple knots s = m - 1
    # https://mdcattaneo.github.io/papers/Cattaneo-Farrell-Feng_2020_AoS--Supplemental.pdf
    if knots is None:
        knots = construct_uniform_knots(p, s, J)
    basis = []
    supports = []
    K_s = calc_K_s(p, s, J)
    for start_knot in range(0, K_s):
        end_knot = start_knot + p + 2  # not included
        basis.append(BSpline.basis_element(knots[start_knot:end_knot], extrapolate=False))
        supports.append((knots[start_knot], knots[end_knot - 1]))

    basis_modified = [compose2(np.nan_to_num, el) for el in basis]

    return basis_modified, supports


def construct_tensor_spline_basis(
    p: Sequence[int],
    s: Sequence[int],
    J: Sequence[int],
    knots_list: Sequence[np.ndarray] | None = None,
) -> tuple[list[Callable[[np.ndarray], float]], List[Tuple[Tuple[float, float], ...]]]:
    """
    Build a tensor-product B-spline basis on a Cartesian grid.

    Parameters
    ----------
    p, s, J : sequences of length D
        Degree, smoothness and resolution per dimension.
    knots_list : list[np.ndarray] | None
        Pre-computed knot vectors per dimension (optional).
    Returns
    -------
    basis  : list[Callable[[np.ndarray], float]]
        Flat list of tensor-product basis functions.
    supports : list[Tuple[(float,float), …]]
        Coordinate-wise closed support for every basis element.
    """
    p, s, J = map(list, (p, s, J))
    D = len(p)
    if not (len(s) == len(J) == D):
        raise ValueError("p, s and J must have the same length")

    # 1-D bases per dimension -------------------------------------------------
    if knots_list is None:
        knots_list = [
            construct_uniform_knots(p[d], s[d], J[d]) for d in range(D)
        ]

    basis_per_dim: list[list[Callable[[float], float]]] = []
    supports_per_dim: list[list[Tuple[float, float]]] = []

    for d in range(D):
        b, sup = construct_spline_basis(
            p[d], s[d], J[d], knots_list[d],
        )
        basis_per_dim.append(b)
        supports_per_dim.append(sup)

    # tensor-product basis ----------------------------------------------------
    tensor_basis: list[Callable[[np.ndarray], float]] = []
    tensor_supports: list[Tuple[Tuple[float, float], ...]] = []

    # Cartesian product of 1-D indices, e.g. (i, j, k)
    for multi_idx in itertools.product(
        *[range(len(basis_per_dim[d])) for d in range(D)]
    ):
        factors = [basis_per_dim[d][i] for d, i in enumerate(multi_idx)]
        supports = tuple(
            supports_per_dim[d][i] for d, i in enumerate(multi_idx)
        )

        # a tiny closure factory avoids late-binding pitfalls
        def make_tensor(factors=factors) -> Callable[[np.ndarray], float]:
            def _tensor(x: np.ndarray) -> float:
                # expects x.shape == (D,)
                x = np.asarray(x, dtype=float).reshape(-1)
                assert x.shape == (D,)
                val = 1.0
                for d, f in enumerate(factors):
                    val *= f(x[d])
                return val

            return _tensor

        tensor_basis.append(make_tensor())
        tensor_supports.append(supports)

    return tensor_basis, tensor_supports


def generate_sample(n, d, model_num, seed=None):
    rng = np.random.default_rng(seed)

    def m0(x):
        return np.sin(2 * np.pi * x).sum(axis=1) / (2 * d)

    def m1(x):
        if d == 1:
            return np.sin(5 * x[:, 0]) * np.sin(10 * x[:, 0]) / 2
        if d == 2:
            return np.sin(5 * x[:, 0]) * np.sin(10 * x[:, 1]) / 2
        raise ValueError(f"d = {d} is not supported")

    def m2(x):
        if d == 1:
            return (1 - (4 * x[:, 0] - 2)**2)**2 / 4
        if d == 2:
            return (1 - (4 * x[:, 0] - 2)**2)**2 * np.sin(5 * x[:, 1]) / 4
        raise ValueError(f"d = {d} is not supported")

    if model_num in (0, 1, 2):
        X = rng.uniform(0.0, 1.0, size=(n, d))            # covariates
        funcs = [m0, m1, m2]
        m = funcs[model_num]

        eps = rng.normal(size=n)
        Y = m(X) + eps                                    # responses

        # conditional CDF  F_{Y|X}(q|x)
        def F_cdf(q: float, x: np.ndarray) -> float | np.ndarray:
            x_arr = np.asarray(x, dtype=float)
            return norm.cdf(q - m(x_arr))

        return Y, X, F_cdf, Link("log-log")

    if model_num == 3:
        if d != 1:
            raise ValueError("model_num = 3 is defined only for d = 1")

        # Generate X ∼ 0.3 + 0.4 U_x  (so X ∈ [0.3,0.7])
        U_x = rng.uniform(size=(n, 1))
        # X   = 0.3 + 0.4 * U_x              # shape (n,1)
        X = U_x  # the support needs to be [0.0, 1.0]

        # Generate treatment T | X
        U_t = rng.uniform(size=n)
        T   = (U_t < X[:, 0]).astype(float)   # 0/1

        # Potential outcomes
        U_y0 = rng.uniform(size=n)
        U_y1 = rng.uniform(size=n)

        # Y(0)
        mask0 = U_y0 <= X[:, 0]
        Y0 = np.where(mask0,
                      U_y0**2 / X[:, 0],          # U ≤ X
                      U_y0)                       # U > X

        # Y(1)
        mask1 = U_y1 <= (1.0 - X[:, 0])
        Y1 = np.where(mask1,
                      U_y1**2 / (1.0 - X[:, 0]),
                      U_y1)

        # Observed outcome
        Y = T * Y1 + (1 - T) * Y0

        # -------- conditional CDF -------------------------------------
        def F_cdf(q: float, x_raw: np.ndarray) -> np.ndarray:
            """
            F_{Y|X}(q | x) = (1-x)·F0(q|x) + x·F1(q|x).
            Works for scalar q and x_raw of shape (k,) or (k,1).
            """
            x = np.asarray(x_raw, dtype=float).reshape(-1)
            q = float(q)

            # helper: piece-wise CDF for Y0 | X = x
            def F0(q_val, xi):
                if q_val < 0:
                    return 0.0
                if q_val < xi:
                    return np.sqrt(xi * q_val)
                if q_val <= 1.0:
                    return q_val
                return 1.0

            # helper: piece-wise CDF for Y1 | X = x
            def F1(q_val, xi):
                if q_val < 0:
                    return 0.0
                if q_val < 1.0 - xi:
                    return np.sqrt((1.0 - xi) * q_val)
                if q_val <= 1.0:
                    return q_val
                return 1.0

            F_vals = np.empty_like(x)
            for idx, xi in enumerate(x):
                F0_val = F0(q, xi)
                F1_val = F1(q, xi)
                F_vals[idx] = (1.0 - xi) * F0_val + xi * F1_val
            return F_vals

        return Y, X, F_cdf, Link("log-log")

    raise ValueError(f"model_num = {model_num} is not supported")


@timing
def calc_betahat_dict(grid_q, P_X, Y, X, link):
    beta_hat_dict = {}
    for q in grid_q:
        beta_hat, _ = optimise_beta(Y, X, P_X, q=q, link=link)
        beta_hat_dict[q] = beta_hat
    return beta_hat_dict


@timing
def calc_matrices(grid_q, basis, P_X, X, F_cond, beta_hat_dict, link, compute_bar=False):
    n = len(X)
    Qbar_dict = {}
    Qhat_dict = {}
    Sigmahat_dict = {}
    Sigmabar_dict = {}
    Omegabar_dict = {}
    Omegahat_dict = {}
    mu_hat_dict = {}
    eta_hat_dict = {}
    d_eta_hat_dict = {}

    Omegahat_inf_norm = 0.0
    Qbar_inv_dict = dict()
    Qhat_inv_dict = dict()
    for q in grid_q:
        mu_hat_dict[q] = P_X @ beta_hat_dict[q]
        eta_hat_dict[q] = link(mu_hat_dict[q])
        d_eta_hat_dict[q] = link.derivative(mu_hat_dict[q])
        if compute_bar:
            Qbar_dict[q] = compute_qbar(X, q, P_X, F_cond, link)
            Qbar_inv_dict[q] = np.linalg.inv(Qbar_dict[q])
        Qhat_dict[q] = compute_qhat(P_X, beta_hat_dict[q], link)
        Qhat_inv_dict[q] = np.linalg.inv(Qhat_dict[q])
        if compute_bar:
            Sigmabar_dict[q] = compute_sigmabar(X, q, P_X, F_cond, link)
        Sigmahat_dict[q] = compute_sigmahat(P_X, beta_hat_dict[q], link)
        if compute_bar:
            Omegabar_dict[q] = make_omegahat(basis, Qbar_inv_dict[q], Sigmabar_dict[q])  # same function as Omegahat
        Omegahat_dict[q] = make_omegahat(basis, Qhat_inv_dict[q], Sigmahat_dict[q])  # function of x
        tmp_omegahat_vector = Omegahat_dict[q](X)
        Omegahat_inf_norm = max(np.linalg.norm(tmp_omegahat_vector / n, ord=np.inf), Omegahat_inf_norm)

    logging.info(f"Omegahat(q, X) / n inf norm = {Omegahat_inf_norm}")

    return mu_hat_dict, d_eta_hat_dict, Qbar_inv_dict, Qhat_inv_dict, Omegabar_dict, Omegahat_dict


def calc_linear_coefficients(grid_x, basis, grid_q, Qbar_inv_dict, Qhat_inv_dict, Omegabar_map, Omegahat_map, compute_bar=False):
    N_x = len(grid_x)
    K = len(basis)
    Pgrid  = make_design_matrix(grid_x, basis)
    C_q_bar_sqrtOm_dict = {}
    C_q_hat_sqrtOm_dict = {}
    for q in grid_q:
        if compute_bar:
            C_q_bar = np.einsum('nk,kl->nl', Pgrid, Qbar_inv_dict[q])
            sqrtOmegabar_q = np.sqrt(np.maximum(Omegabar_map(q, grid_x), 1e-12))
            C_q_bar_sqrtOm_dict[q] = C_q_bar / sqrtOmegabar_q[:, None]
        else:
            C_q_bar_sqrtOm_dict[q] = None

        C_q_hat = np.einsum('nk,kl->nl', Pgrid, Qhat_inv_dict[q])
        sqrtOmegahat_q = np.sqrt(np.maximum(Omegahat_map(q, grid_x), 1e-12))
        C_q_hat_sqrtOm_dict[q] = C_q_hat / sqrtOmegahat_q[:, None]

        assert C_q_hat_sqrtOm_dict[q].shape == (N_x, K)

    return C_q_bar_sqrtOm_dict, C_q_hat_sqrtOm_dict


def closest_psd_matrix(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


@timing
def check_func_approx(P_grid, grid_x, grid_q, beta_hat_dict, F_cond, link):
    maximal_uniform_diff = 0.0
    maximal_rmse_diff = 0.0
    for q in grid_q:
        uniform_diff = np.linalg.norm(P_grid @ beta_hat_dict[q] - link.inverse(F_cond(q, grid_x)), ord=np.inf)
        rmse_diff = np.sqrt(np.mean((P_grid @ beta_hat_dict[q] - link.inverse(F_cond(q, grid_x)))**2))
        maximal_uniform_diff = max(uniform_diff, maximal_uniform_diff)
        maximal_rmse_diff = max(rmse_diff, maximal_rmse_diff)

    return maximal_uniform_diff, maximal_rmse_diff


@timing
def construct_gaussian_field(
        basis, grid_x, grid_q,
        Qbar_inv_dict, Qhat_inv_dict,
        Sigmabar_map, Sigmahat_map,
        Omegabar_map, Omegahat_map,
        seed=None, compute_bar=False,
    ):
    if compute_bar:
        sigmabar_block_matrix = build_sigmahat_block_matrix(grid_q, len(basis), Sigmabar_map)
    else:
        sigmabar_block_matrix = None
    sigmahat_block_matrix = build_sigmahat_block_matrix(grid_q, len(basis), Sigmahat_map)

    logging.info(f"lambda_min(sigmahat_block_matrix), lambda_max(sigmahat_block_matrix) = {np.linalg.eigvals(sigmahat_block_matrix).min()}, {np.linalg.eigvals(sigmahat_block_matrix).max()}")

    num_of_exps_for_Z = 10000

    # Construct a K N_q dimensional Gaussian vector with this covariance matrix
    rng = np.random.default_rng(seed)
    if compute_bar:
        sigmabar_block_matrix_cholesky = cholesky(sigmabar_block_matrix, lower=True)
    else:
        sigmabar_block_matrix_cholesky = None
    sigmahat_block_matrix = closest_psd_matrix(sigmahat_block_matrix)
    sigmahat_block_matrix_cholesky = cholesky(sigmahat_block_matrix, lower=True)
    standard_normal_K = rng.standard_normal((sigmahat_block_matrix.shape[0], num_of_exps_for_Z))
    if compute_bar:
        Zbar_K = sigmabar_block_matrix_cholesky @ standard_normal_K
    Zhat_K = sigmahat_block_matrix_cholesky @ standard_normal_K

    C_q_bar_sqrtOm_dict, C_q_hat_sqrtOm_dict = calc_linear_coefficients(grid_x, basis, grid_q, Qbar_inv_dict, Qhat_inv_dict, Omegabar_map, Omegahat_map)

    if compute_bar:
        Zbar_field = construct_z_fields(Zbar_K, C_q_bar_sqrtOm_dict, grid_q)
    else:
        Zbar_field = None
    Zhat_field = construct_z_fields(Zhat_K, C_q_hat_sqrtOm_dict, grid_q)
    assert Zhat_field.shape == (len(grid_x) * len(grid_q), num_of_exps_for_Z)

    return Zbar_field, Zhat_field


def calc_quantiles(quantile, Zbar_field, Zhat_field, compute_bar=False):
    # Calculate the quantile
    if compute_bar:
        max_Zbar = np.max(np.abs(Zbar_field), axis=0)    # shape (num_of_exps_for_Z,)
    else:
        max_Zbar = None
    max_Zhat = np.max(np.abs(Zhat_field), axis=0)

    if compute_bar:
        quantile_Zbar = np.quantile(max_Zbar, quantile)
    else:
        quantile_Zbar = None

    quantile_Zhat = np.quantile(max_Zhat, quantile)
    logging.info(f"quantile_Zbar = {quantile_Zbar}, quantile_Zhat = {quantile_Zhat}")

    return quantile_Zbar, quantile_Zhat


@timing
def check_approx_and_coverage(basis, Y, X, F_cond, grid_q, grid_x, link, seed=None):
    n, d = X.shape
    P_X = make_design_matrix(X, basis)

    beta_hat_dict = calc_betahat_dict(grid_q, P_X, Y, X, link)

    P_grid = make_design_matrix(grid_x, basis)
    maximal_uniform_diff, maximal_rmse_diff = check_func_approx(P_grid, grid_x, grid_q, beta_hat_dict, F_cond, link)
    logging.info(f"inf_norm_diff = {maximal_uniform_diff}")
    logging.info(f"maximal rmse diff = {maximal_rmse_diff}")

    F_vals = np.array([F_cond(q, X) for q in grid_q])
    logging.info(f"min F_cond(q, X) over all q = {F_vals.min()}, max F_cond(q, X) over all q = {F_vals.max()}")

    (
        mu_hat_dict, d_eta_hat_dict,
        Qbar_inv_dict, Qhat_inv_dict,
        Omegabar_dict, Omegahat_dict,
    ) = calc_matrices(grid_q, basis, P_X, X, F_cond, beta_hat_dict, link)

    def Omegabar_map(q, x):
        return Omegabar_dict[q](x)

    def Omegahat_map(q, x):
        return Omegahat_dict[q](x)

    def Sigmabar_map(q1, q2):
        S = compute_sigmabar_q1_q2(q1, q2, F_cond, X, P_X, link)
        return 0.5 * (S + S.T)

    def Sigmahat_map(q1, q2):
        S = compute_sigmahat_q1_q2(q1, q2, mu_hat_dict, d_eta_hat_dict, P_X, link)
        return 0.5 * (S + S.T)

    Zbar_field, Zhat_field = construct_gaussian_field(
        basis, grid_x, grid_q,
        Qbar_inv_dict, Qhat_inv_dict,
        Sigmabar_map, Sigmahat_map,
        Omegabar_map, Omegahat_map,
        seed=seed,
    )

    quantile_Zbar, quantile_Zhat = calc_quantiles(0.95, Zbar_field, Zhat_field)

    # Check coverage on the grid
    # use different grid?
    root_Omegahat_inf_norm = 0.0
    coverage_for_qs = []
    confidence_band_width = 0.0
    average_on_grid_cb_width = 0.0
    for q in grid_q:
        mu_hat = P_grid @ beta_hat_dict[q]
        rescaled_quantile = quantile_Zhat * np.sqrt(Omegahat_map(q, grid_x) / n)
        root_Omegahat_inf_norm = max(np.linalg.norm(np.sqrt(Omegahat_map(q, grid_x) / n), ord=np.inf), root_Omegahat_inf_norm)
        left_end = mu_hat - rescaled_quantile
        right_end = mu_hat + rescaled_quantile
        confidence_band_width = max(confidence_band_width, np.max(np.abs(right_end - left_end)))
        average_on_grid_cb_width += np.mean(np.abs(right_end - left_end))
        eta_left_end = link(left_end)
        eta_right_end = link(right_end)
        does_cover_for_this_q = np.all((F_cond(q, grid_x) > eta_left_end) & (F_cond(q, grid_x) < eta_right_end))
        coverage_for_qs.append(does_cover_for_this_q)

    average_on_grid_cb_width /= len(grid_q)
    uniform_coverage_bool = np.all(coverage_for_qs)
    logging.info(f"root Omegahat(q, grid_x) / n inf norm = {root_Omegahat_inf_norm}")

    return maximal_uniform_diff, maximal_rmse_diff, uniform_coverage_bool, confidence_band_width, average_on_grid_cb_width


def run_monte_carlo(n, d, model_num, grid_q, grid_x, knots_per_dim=4, num_of_exps=100):
    p = [3] * d
    s = [2] * d
    J = [knots_per_dim] * d

    basis, _ = construct_tensor_spline_basis(p, s, J)
    logging.info(f"n = {n}, d = {d}, p = {p}, s = {s}, knots_per_dim = {knots_per_dim}, model_num = {model_num}, constructed K = {len(basis)} basis functions")

    pointwise_approx_lists = [[], [], []]
    pointwise_coverage_bool_lists = [[], [], []]
    pointwise_confidence_interval_width_lists = [[], [], []]
    uniform_coverage_bool_list = []
    confidence_band_width_list = []
    average_on_grid_cb_width_list = []
    uniform_approx_diff_list = []
    rmse_diff_list = []

    for exp_idx in range(num_of_exps):
        for attempt_idx in range(5):
            try:
                start_time = time.time()
                logging.info(f"exp_idx = {exp_idx}")
                Y, X, F_cond, link = generate_sample(n=n, d=d, model_num=model_num, seed=None)

                if grid_q[0] < 0.0:
                    q_points = [-0.2, 0.0, 0.2]
                else:
                    q_points = [0.45, 0.6, 0.75]

                if d == 1:
                    x_points = [[0.3], [0.1], [0.2]]
                elif d == 2:
                    x_points = [[0.3, 0.1], [0.1, 0.4], [0.2, 0.2]]
                else:
                    raise ValueError(f"d = {d} is not supported")

                new_pointwise_approx_lists = [[], [], []]
                new_pointwise_coverage_bool_lists = [[], [], []]
                new_pointwise_confidence_interval_width_lists = [[], [], []]
                for point_idx, (q_point, x_point) in enumerate(zip(q_points, x_points)):
                    (
                        maximal_uniform_diff, maximal_rmse_diff,
                        pointwise_coverage_bool, confidence_interval_width,
                        _,
                    ) = check_approx_and_coverage(
                        basis, Y, X, F_cond,
                        np.array([q_point]), np.array([x_point]), link, seed=None,
                    )
                    new_pointwise_approx_lists[point_idx].append(maximal_uniform_diff)
                    new_pointwise_coverage_bool_lists[point_idx].append(pointwise_coverage_bool)
                    new_pointwise_confidence_interval_width_lists[point_idx].append(confidence_interval_width)

                (
                    maximal_uniform_diff, maximal_rmse_diff,
                    uniform_coverage_bool, confidence_band_width,
                    average_on_grid_cb_width,
                ) = check_approx_and_coverage(basis, Y, X, F_cond, grid_q, grid_x, link, seed=None)

                # extend the lists once, if no exception was raised
                for point_idx in range(len(q_points)):
                    pointwise_approx_lists[point_idx].extend(new_pointwise_approx_lists[point_idx])
                    pointwise_coverage_bool_lists[point_idx].extend(new_pointwise_coverage_bool_lists[point_idx])
                    pointwise_confidence_interval_width_lists[point_idx].extend(new_pointwise_confidence_interval_width_lists[point_idx])

                uniform_approx_diff_list.append(maximal_uniform_diff)
                rmse_diff_list.append(maximal_rmse_diff)
                confidence_band_width_list.append(confidence_band_width)
                average_on_grid_cb_width_list.append(average_on_grid_cb_width)
                uniform_coverage_bool_list.append(uniform_coverage_bool)

                logging.info(f"uniform coverage = {uniform_coverage_bool}")
                logging.info(f"maximal_uniform_diff = {maximal_uniform_diff}")
                logging.info(f"maximal_rmse_diff = {maximal_rmse_diff}")
                logging.info(f"confidence_band_width = {confidence_band_width}")
                logging.info(f"average_on_grid_cb_width = {average_on_grid_cb_width}")
                logging.info(f"this exp took {time.time() - start_time} seconds")
                logging.info('\n\n\n\n')
                break
            except Exception as e:
                if attempt_idx == 4:
                    raise e
                logging.error(f"attempt {attempt_idx} failed, retrying...")
                continue

    logging.info(f"pointwise_approx_lists means = {np.mean(pointwise_approx_lists, axis=1)}")  # shape (2,)
    logging.info(f"pointwise_coverage_bool_lists means = {np.mean(pointwise_coverage_bool_lists, axis=1)}")  # shape (2,)
    logging.info(f"uniform_coverage_bool_list mean = {np.mean(uniform_coverage_bool_list)}")
    logging.info(f"pointwise_confidence_interval_width_lists means = {np.mean(pointwise_confidence_interval_width_lists, axis=1)}")  # shape (2,)
    logging.info(f"confidence_band_width_list mean = {np.mean(confidence_band_width_list)}")
    logging.info(f"average_on_grid_cb_width_list mean = {np.mean(average_on_grid_cb_width_list)}")
    logging.info(f"uniform_approx_diff_list mean = {np.mean(uniform_approx_diff_list)}")
    logging.info(f"rmse_diff_list mean = {np.mean(rmse_diff_list)}")


    # save results
    pointwise_coverage_bool_lists = [[int(x) for x in sublist] for sublist in pointwise_coverage_bool_lists]
    uniform_coverage_bool_list = [int(x) for x in uniform_coverage_bool_list]

    results = {
        "pointwise_approx_lists": pointwise_approx_lists,
        "pointwise_coverage_bool_lists": pointwise_coverage_bool_lists,
        "uniform_coverage_bool_list": uniform_coverage_bool_list,
        "pointwise_confidence_interval_width_lists": pointwise_confidence_interval_width_lists,
        "confidence_band_width_list": confidence_band_width_list,
        "average_on_grid_cb_width_list": average_on_grid_cb_width_list,
        "uniform_approx_diff_list": uniform_approx_diff_list,
        "rmse_diff_list": rmse_diff_list,
    }

    return results


def work(exp_number, iter_number):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    worker_num = int(os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
    job_id = os.environ.get("SLURM_JOB_ID")

    logging.warning(
        "Exp%d, iter%d, job_id: %s, worker_num: %d",
        exp_number,
        iter_number,
        job_id,
        worker_num,
    )

    args = get_args(exp_number)[worker_num]

    results_dct = run_monte_carlo(**args)

    results_dir = f"results/exp{exp_number:02d}_iter{iter_number:02d}"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/model{args['model_num']}_knots{args['knots_per_dim']}.json", "w") as f:
        json.dump(results_dct, f)


if __name__ == "__main__":
    work(0, 0)
