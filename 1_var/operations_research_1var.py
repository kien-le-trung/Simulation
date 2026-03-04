from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel


# ----------------------------
# Candidate lattice (distance only, fixed time)
# ----------------------------
D_MIN, D_MAX = 0.05, 1.0
T_FIXED_DEFAULT = 5.0

D_STEP = 0.05


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


# ----------------------------
# Pre-trial predictor in SPEED domain
# ----------------------------
def p_hit_from_speed(v_req, v_hat, sigma_v):
    """
    Predict P(hit) from speed model:
        v_eff ~ N(v_hat, sigma_v^2)
        hit if v_eff >= v_req
    => P(hit) = 1 - Phi((v_req - v_hat)/sigma_v)
    """
    if sigma_v < 1e-6:
        return 1.0 if v_hat >= v_req else 0.0
    z = (v_req - v_hat) / sigma_v
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


# ----------------------------
# Map continuous d to PatientModel distance_level
# ----------------------------
def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    candidates = np.where(patient.d_means <= d_sys)[0]
    if len(candidates) == 0:
        return 0
    return int(candidates[-1])


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


def expand_bounds_if_needed(d_min, d_max, observed_speeds, t_fixed):
    """
    Backup plan for distance-only bounds based on observed speed envelope.
    Never shrinks. Returns (d_min, d_max, changed).
    """
    abs_d_max = 1.5

    if len(observed_speeds) < 5:
        return d_min, d_max, False

    arr = np.array(observed_speeds[-50:])
    v_p95 = float(np.percentile(arr, 95))
    d_p95 = v_p95 * max(float(t_fixed), 1e-6)

    changed = False
    new_d_max = min(abs_d_max, max(d_max, d_p95 * 1.2))
    if new_d_max > d_max * 1.05:
        d_max = new_d_max
        changed = True

    return d_min, d_max, changed


def make_rom_state(*, d_min: float = D_MIN, d_max: float = D_MAX, n_bins: int = 12):
    edges = np.linspace(float(d_min), float(d_max), int(n_bins) + 1)
    shape = (int(n_bins),)
    return {
        "distance_edges": edges,
        "hit_counts": np.zeros(shape, dtype=float),
        "miss_counts": np.zeros(shape, dtype=float),
        "sum_dist_ratio_on_miss": np.zeros(shape, dtype=float),
    }


def rom_bin_index(d_sys: float, distance_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(distance_edges, float(d_sys), side="right") - 1)
    return int(np.clip(idx, 0, len(distance_edges) - 2))


def update_rom_state(rom_state: dict, *, d_sys: float, hit: bool, dist_ratio: float):
    d_edges = np.asarray(rom_state["distance_edges"], dtype=float)
    b = rom_bin_index(d_sys, d_edges)
    if hit:
        rom_state["hit_counts"][b] += 1.0
    else:
        rom_state["miss_counts"][b] += 1.0
        rom_state["sum_dist_ratio_on_miss"][b] += float(np.clip(dist_ratio, 0.0, 1.0))


def rom_penalty_term(
    d_sys: float,
    *,
    rom_state: dict | None,
    trial_index: int,
    warmup_trials: int = 40,
    w_rom: float = 0.45,
    tau_d: float = 0.05,
    boundary_threshold: float = 0.35,
):
    # Return additive objective penalty learned from miss patterns at larger distances.
    # No oracle max-reach is used; this is derived only from observed trial outcomes.
    if rom_state is None:
        return 0.0, 0.0, 0.0, float(D_MAX)

    hit_counts = np.asarray(rom_state["hit_counts"], dtype=float)
    miss_counts = np.asarray(rom_state["miss_counts"], dtype=float)
    miss_ratio_sum = np.asarray(rom_state["sum_dist_ratio_on_miss"], dtype=float)
    d_edges = np.asarray(rom_state["distance_edges"], dtype=float)
    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])

    h = hit_counts
    m = miss_counts
    n = h + m

    miss_rate = (m + 1.0) / (n + 2.0)
    miss_severity = (m - miss_ratio_sum + 1.0) / (m + 2.0)
    rom_evidence = miss_rate * miss_severity
    support = np.clip(n / 8.0, 0.0, 1.0)
    weighted_evidence = rom_evidence * support

    if np.any(weighted_evidence > boundary_threshold):
        boundary_idx = int(np.argmax(weighted_evidence > boundary_threshold))
        d_boundary = float(d_centers[boundary_idx])
    else:
        # If no boundary is discovered yet, keep penalty weak at long distances only.
        d_boundary = float(d_edges[-1])

    dist_risk = 1.0 / (1.0 + math.exp(-(float(d_sys) - d_boundary) / max(float(tau_d), 1e-6)))
    conf = float(np.clip(np.sum(n) / 24.0, 0.0, 1.0))
    warmup = float(np.clip(float(trial_index) / max(float(warmup_trials), 1.0), 0.0, 1.0))
    tail_mask = d_centers >= float(d_sys)
    if np.any(tail_mask):
        tail_severity = float(np.mean(weighted_evidence[tail_mask]))
    else:
        tail_severity = float(np.mean(weighted_evidence))
    tail_severity = float(np.clip(tail_severity, 0.0, 1.0))

    risk = float(np.clip(dist_risk * conf * max(tail_severity, 0.25), 0.0, 1.0))
    penalty = float(w_rom * warmup * risk)
    return penalty, risk, conf, d_boundary


# ----------------------------
# Objective in speed domain
# ----------------------------
def score_speed_candidate(
    v_req,
    *,
    d_sys,
    v_hat,
    sigma_v,
    p_star,
    p_min=0.05,
    rom_state: dict | None = None,
    trial_index: int = 0,
    rom_warmup_trials: int = 20,
    w_rom: float = 0.45,
    rom_tau_d: float = 0.05,
    rom_boundary_threshold: float = 0.35,
):
    p = p_hit_from_speed(v_req, v_hat, sigma_v)

    if p < p_min:
        return -1e9, p, 0.0, 0.0, 0.0, float(D_MAX)

    eff_raw = (p - p_star) ** 2
    eff_normalized = 1.0 - eff_raw / (p_star ** 2)
    score = eff_normalized
    rom_penalty, rom_risk, rom_confidence, rom_boundary = rom_penalty_term(
        d_sys,
        rom_state=rom_state,
        trial_index=trial_index,
        warmup_trials=rom_warmup_trials,
        w_rom=w_rom,
        tau_d=rom_tau_d,
        boundary_threshold=rom_boundary_threshold,
    )
    score = score - rom_penalty
    return score, p, rom_penalty, rom_risk, rom_confidence, rom_boundary


# ----------------------------
# Main loop
# ----------------------------
def run_sim(
    patient: PatientModel,
    n_trials=10000,
    seed=7,
    ema_alpha=0.20,
    calibration=True,
    t_fixed=T_FIXED_DEFAULT,
):
    t_fixed = float(t_fixed)

    cur_d_min, cur_d_max = D_MIN, D_MAX

    p_star = 0.70
    p_min = 0.05

    counts_5x5 = np.zeros((5, 5), dtype=int)
    previous_hit = True

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        cur_d_min, cur_d_max = derive_bounds_from_calibration(calibration_result, patient)

    local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)

    # Online speed model (EWMA + EW second moment).
    sigma_v_floor = 0.02
    if calibration_result:
        v_obs_init = []
        for trial in calibration_result.get("trials", []):
            d_sys = float(trial.get("d_sys", 0.0))
            t_cap = float(trial.get("t_cap", t_fixed))
            reached = bool(trial.get("reached", trial.get("hit", False)))
            if reached:
                t_pat_obs = float(trial.get("t_pat_obs", trial.get("t_pat", t_cap)))
                v_obs = d_sys / max(t_pat_obs, 1e-6)
            else:
                d_pat = float(trial.get("d_pat", float(trial.get("dist_ratio", 0.0)) * d_sys))
                v_obs = d_pat / max(t_cap, 1e-6)
            v_obs_init.append(v_obs)

        if len(v_obs_init) > 0:
            v_hat = float(np.mean(v_obs_init))
            sigma_v = float(max(np.std(v_obs_init), sigma_v_floor))
        else:
            v_hat = 0.60
            sigma_v = 0.25
    else:
        v_hat = 0.60
        sigma_v = 0.25

    m2_hat = v_hat**2 + sigma_v**2

    hist = {
        "d": [],
        "t": [],
        "p_pred": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "v_hat": [],
        "sigma_v": [],
        "score": [],
        "rom_penalty": [],
        "rom_risk": [],
        "rom_confidence": [],
        "rom_boundary": [],
    }
    observed_speeds = []
    rom_state = make_rom_state(d_min=cur_d_min, d_max=cur_d_max)

    for k in range(n_trials):
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_d_min, new_d_max, changed = expand_bounds_if_needed(
                cur_d_min,
                cur_d_max,
                observed_speeds,
                t_fixed=t_fixed,
            )
            if changed:
                cur_d_min, cur_d_max = new_d_min, new_d_max
                local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)

        best_d = None
        best_score = -1e18
        best_p = None
        best_rom_penalty = 0.0
        best_rom_risk = 0.0
        best_rom_conf = 0.0
        best_rom_boundary = float(D_MAX)

        for d_cand in local_d_grid:
            v_req = float(d_cand) / max(t_fixed, 1e-9)
            sc, p, rom_penalty, rom_risk, rom_conf, rom_boundary = score_speed_candidate(
                v_req,
                d_sys=float(d_cand),
                v_hat=v_hat,
                sigma_v=sigma_v,
                p_star=p_star,
                p_min=p_min,
                rom_state=rom_state,
                trial_index=k,
                rom_warmup_trials=20,
                w_rom=0.45,
                rom_tau_d=0.05,
                rom_boundary_threshold=0.35,
            )
            if sc > best_score:
                best_score = sc
                best_d = float(d_cand)
                best_p = p
                best_rom_penalty = float(rom_penalty)
                best_rom_risk = float(rom_risk)
                best_rom_conf = float(rom_conf)
                best_rom_boundary = float(rom_boundary)

        d_sys = float(np.clip(best_d, cur_d_min, cur_d_max))
        t_sys = t_fixed

        i = level5(d_sys, cur_d_min, cur_d_max)
        j = 0
        counts_5x5[i, j] += 1

        lvl = distance_level_from_patient_bins(patient, d_sys)
        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
        )
        hit = bool(outcome["hit"])
        previous_hit = hit
        update_rom_state(
            rom_state,
            d_sys=float(d_sys),
            hit=hit,
            dist_ratio=float(outcome["dist_ratio"]),
        )

        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])
        if hit and t_pat > 0.01:
            v_obs = d_pat / max(t_pat, 1e-6)
        else:
            v_obs = d_pat / max(t_sys, 1e-6)
        observed_speeds.append(v_obs)

        alpha = float(np.clip(ema_alpha, 1e-6, 1.0))
        v_hat = (1.0 - alpha) * v_hat + alpha * v_obs
        m2_hat = (1.0 - alpha) * m2_hat + alpha * (v_obs ** 2)
        sigma_v = float(np.sqrt(max(m2_hat - v_hat**2, sigma_v_floor**2)))

        hist["d"].append(d_sys)
        hist["t"].append(t_sys)
        hist["p_pred"].append(best_p)
        hist["hit"].append(int(hit))
        hist["time_ratio"].append(float(outcome["time_ratio"]))
        hist["dist_ratio"].append(float(outcome["dist_ratio"]))
        hist["v_hat"].append(float(v_hat))
        hist["sigma_v"].append(float(sigma_v))
        hist["score"].append(best_score)
        hist["rom_penalty"].append(best_rom_penalty)
        hist["rom_risk"].append(best_rom_risk)
        hist["rom_confidence"].append(best_rom_conf)
        hist["rom_boundary"].append(best_rom_boundary)

    return hist, counts_5x5, patient
