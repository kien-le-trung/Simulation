from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

D_MIN, D_MAX = 0.10, 0.80
T_FIXED_DEFAULT = 5.0


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_sim = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_sim.PatientModel


def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def entropy(posterior, eps=1e-12):
    p = posterior.ravel().astype(float)
    p = p / (p.sum() + eps)
    p = p[p > 0]
    return -np.sum(p * np.log(p + eps))


def normalize(p, eps=1e-12):
    s = np.sum(p)
    if s <= eps:
        return np.full_like(p, 1.0 / p.size)
    return p / s


def cartesian_params(b0_grid, b1_grid):
    b0 = b0_grid[:, None]
    b1 = b1_grid[None, :]
    return b0, b1


def p_hit_model(d, b0, b1):
    return sigmoid(b0 + b1 * d)


def expected_entropy_for_x(d, posterior, b0, b1, eps=1e-12):
    p_hit = p_hit_model(d, b0, b1)
    p_miss = 1.0 - p_hit

    p_hit_pred = float(np.sum(posterior * p_hit))
    p_miss_pred = 1.0 - p_hit_pred

    post_hit = normalize(posterior * p_hit, eps=eps)
    post_miss = normalize(posterior * p_miss, eps=eps)

    h_hit = entropy(post_hit, eps=eps)
    h_miss = entropy(post_miss, eps=eps)

    eh = p_hit_pred * h_hit + p_miss_pred * h_miss
    return eh, p_hit_pred


def map_distance_level(patient: PatientModel, d_sys: float) -> int:
    d_means = np.asarray(patient.d_means, dtype=float)
    idxs = np.where(d_means <= d_sys)[0]
    if len(idxs) == 0:
        return 0
    return int(idxs[-1])


def level5(x, xmin, xmax):
    u = (x - xmin) / (xmax - xmin + 1e-12)
    if u < 0.2:
        return 0
    if u < 0.4:
        return 1
    if u < 0.6:
        return 2
    if u < 0.8:
        return 3
    return 4


def apply_calibration_priors(patient: PatientModel, calibration_result: dict | None):
    if not calibration_result:
        return
    per_direction = calibration_result.get("per_direction", {})
    for direction, stats in per_direction.items():
        idx = int(np.clip(direction, 0, 4))
        n_reached = float(stats.get("n_reached", 0))
        n_censored = float(stats.get("n_censored", 0))
        patient.spatial_success_alpha[idx] += n_reached
        patient.spatial_success_beta[idx] += n_censored


def derive_bounds_from_calibration(calibration_result, patient):
    abs_d_min, abs_d_max = 0.05, 1.5

    if not calibration_result:
        return D_MIN, D_MAX

    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = abs_d_min

    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15

    return float(d_min_cal), float(min(d_max_cal, abs_d_max))


def build_calibration_prior(calibration_result, posterior_shape, b0, b1, dmin, dmax):
    if not calibration_result:
        return normalize(np.ones(posterior_shape, dtype=float))

    trials = calibration_result.get("trials", [])
    if len(trials) < 3:
        return normalize(np.ones(posterior_shape, dtype=float))

    n_hits = sum(1 for tr in trials if tr.get("hit", tr.get("reached", False)))
    cal_hit_rate = n_hits / max(len(trials), 1)

    d_mid = (dmin + dmax) / 2.0
    p_pred = p_hit_model(d_mid, b0, b1)

    sigma = 0.20
    log_weight = -0.5 * ((p_pred - cal_hit_rate) / sigma) ** 2
    prior = np.exp(log_weight - np.max(log_weight))
    return normalize(prior)


def run_quest_plus_dt(
    patient: PatientModel,
    n_trials=120,
    p_target=0.70,
    p_tol=0.08,
    seed=7,
    d_grid=None,
    t_grid=None,
    b0_grid=None,
    b1_grid=None,
    b2_grid=None,
    t_fixed=T_FIXED_DEFAULT,
    calibration=True,
):
    """
    1-var QUEST+ loop over distance lattice only.
    Time is fixed at t_fixed for all trials.
    """
    rng = np.random.default_rng(seed)
    t_fixed = float(t_fixed)

    counts = np.zeros((5, 5), dtype=int)

    dmin, dmax = D_MIN, D_MAX
    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        dmin, dmax = derive_bounds_from_calibration(calibration_result, patient)

    if d_grid is None:
        d_grid = np.round(np.arange(dmin, dmax + 0.01, 0.1), 4)

    if b0_grid is None:
        b0_grid = np.round(np.linspace(-4.0, 4.0, 7), 4)
    if b1_grid is None:
        b1_grid = np.round(np.linspace(-8.0, -0.5, 7), 4)

    b0, b1 = cartesian_params(b0_grid, b1_grid)
    posterior_shape = (len(b0_grid), len(b1_grid))
    posterior = build_calibration_prior(
        calibration_result, posterior_shape, b0, b1, dmin, dmax
    )
    h_max = np.log(np.prod(posterior_shape))

    hist = {
        "d": [],
        "t": [],
        "p_pred": [],
        "EH": [],
        "hit": [],
        "t_pat": [],
        "d_pat": [],
        "time_ratio": [],
        "dist_ratio": [],
        "H_post": [],
    }

    prev_hit = True
    stimuli = [float(d) for d in d_grid]

    for _k in range(n_trials):
        cand = []
        for d in stimuli:
            eh, p_pred = expected_entropy_for_x(d, posterior, b0, b1)
            cand.append((eh, p_pred, d))

        h_current = entropy(posterior)
        h_ratio = h_current / h_max

        if h_ratio > 0.8:
            near = cand
        elif h_ratio > 0.4:
            near = [x for x in cand if abs(x[1] - p_target) <= p_tol * 2]
        else:
            near = [x for x in cand if abs(x[1] - p_target) <= p_tol]

        chosen = min(near, key=lambda x: x[0]) if len(near) > 0 else min(cand, key=lambda x: x[0])
        eh_best, p_pred_best, d_sys = chosen

        i = level5(d_sys, dmin, dmax)
        j = 0
        counts[i, j] += 1

        lvl = map_distance_level(patient, d_sys)
        out = patient.sample_trial(
            t_sys=t_fixed,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=prev_hit,
        )
        hit = bool(out["hit"])
        prev_hit = hit

        p_hit = p_hit_model(d_sys, b0, b1)
        like = p_hit if hit else (1.0 - p_hit)
        posterior = normalize(posterior * like)

        hist["d"].append(d_sys)
        hist["t"].append(t_fixed)
        hist["p_pred"].append(p_pred_best)
        hist["EH"].append(eh_best)
        hist["hit"].append(int(hit))
        hist["t_pat"].append(float(out["t_pat"]))
        hist["d_pat"].append(float(out["d_pat"]))
        hist["time_ratio"].append(float(out["time_ratio"]))
        hist["dist_ratio"].append(float(out["dist_ratio"]))
        hist["H_post"].append(entropy(posterior))

    return hist, counts