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
# Candidate lattice (d,t,dir)
# ----------------------------
D_MIN, D_MAX = 0.05, 1.0
T_MIN, T_MAX = 0.3, 7.0

# resolution (tune for speed vs granularity)
D_STEP = 0.05   # meters
T_STEP = 0.25   # seconds

d_grid = np.round(np.arange(D_MIN, D_MAX + 1e-9, D_STEP), 4)
t_grid = np.round(np.arange(T_MIN, T_MAX + 1e-9, T_STEP), 4)
N_DIRECTIONS = 5
CANDIDATES = np.array(
    [(d, t, direction) for d in d_grid for t in t_grid for direction in range(N_DIRECTIONS)],
    dtype=float,
)


# ----------------------------
# 25-bin mapping for variability point calculation (5x5 grid)
# ----------------------------
def level5(x, xmin, xmax):
    """
    Map x in [xmin,xmax] -> {0,1,2,3,4} by fifths.
    Bin 0 = shortest/closest (0-20th percentile)
    Bin 1 = short/close (20-40th percentile)
    Bin 2 = medium (40-60th percentile)
    Bin 3 = long/far (60-80th percentile)
    Bin 4 = longest/farthest (80-100th percentile)
    """
    u = (x - xmin) / (xmax - xmin + 1e-12)
    if u < 0.2: return 0
    if u < 0.4: return 1
    if u < 0.6: return 2
    if u < 0.8: return 3
    return 4

def bin25(d, t, d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX):
    return (level5(d, d_min, d_max), level5(t, t_min, t_max))  # (dist_level, time_level)

def rarity_bonus(counts_5x5, d, t, d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX, eps=1e-9):
    """Bonus for under-sampled bins (approx -log frequency)."""
    i, j = bin25(d, t, d_min, d_max, t_min, t_max)
    total = counts_5x5.sum()
    # Laplace smoothing
    freq = (counts_5x5[i, j] + 1.0) / (total + 25.0)
    return -math.log(freq + eps)


def direction_rarity_bonus(counts_dir, direction, eps=1e-9):
    """Bonus for under-sampled directions (same -log frequency as rarity_bonus)."""
    total = counts_dir.sum()
    freq = (counts_dir[direction] + 1.0) / (total + float(len(counts_dir)))
    return -math.log(freq + eps)

def smooth_penalty(d, t, d_prev, t_prev, d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX):
    """Quadratic penalty for jumping in normalized space."""
    if d_prev is None or t_prev is None:
        return 0.0
    dn = (d - d_prev) / (d_max - d_min + 1e-12)
    tn = (t - t_prev) / (t_max - t_min + 1e-12)
    return dn*dn + tn*tn


# ----------------------------
# Simple pre-trial predictor: p_hit(d,t)
# (lightweight, uses online estimate of effective speed)
# ----------------------------
def p_hit_from_speed(d, t, v_hat, sigma_v):
    """
    Predict P(hit) from required speed v_req=d/t using a normal model:
        v_eff ~ N(v_hat, sigma_v^2)
        hit if v_eff >= v_req  (patient can move at least required speed)
    => P(hit) = 1 - Phi((v_req - v_hat)/sigma_v)
    """
    v_req = d / max(t, 1e-9)
    if sigma_v < 1e-6:
        return 1.0 if v_hat >= v_req else 0.0
    z = (v_req - v_hat) / sigma_v
    # 1 - Phi(z) using erf
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def combine_hit_probs_odds(p_dt, p_dir, eps=1e-9):
    p_dt = float(np.clip(p_dt, eps, 1.0 - eps))
    p_dir = float(np.clip(p_dir, eps, 1.0 - eps))
    odds_dt = p_dt / (1.0 - p_dt)
    odds_dir = p_dir / (1.0 - p_dir)
    odds = odds_dt * odds_dir
    return odds / (1.0 + odds)


def beta_mode(alpha, beta):
    alpha = float(alpha)
    beta = float(beta)
    if alpha > 1.0 and beta > 1.0:
        return (alpha - 1.0) / (alpha + beta - 2.0)
    if alpha <= 1.0 and beta > 1.0:
        return 0.0
    if beta <= 1.0 and alpha > 1.0:
        return 1.0
    # Both <= 1.0 gives a bimodal U-shape; fall back to neutral midpoint.
    return 0.5


# ----------------------------
# Map continuous d to your PatientModel distance_level (0..7)
# Uses the rule in your comment: closest bucket with d_mean <= d_sys
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
    """
    Derive generous (d_min, d_max, t_min, t_max) from calibration data.
    Uses wide margins (2x) so the grid is never too confined.
    Falls back to module defaults if calibration data is insufficient.
    """
    ABS_D_MIN, ABS_D_MAX = 0.05, 1.5
    ABS_T_MIN, ABS_T_MAX = 0.15, 15.0

    if not calibration_result:
        return D_MIN, D_MAX, T_MIN, T_MAX

    trials = calibration_result.get("trials", [])
    speeds = []
    for tr in trials:
        hit = tr.get("hit", tr.get("reached", False))
        t_pat = float(tr.get("t_pat_obs", tr.get("t_pat", 0)))
        d_sys = float(tr.get("d_sys", 0))
        if hit and t_pat > 0.01 and d_sys > 0.01:
            speeds.append(d_sys / t_pat)

    if len(speeds) < 3:
        return D_MIN, D_MAX, T_MIN, T_MAX

    speeds = np.array(speeds)
    v_slow = max(float(np.percentile(speeds, 10)), 0.01)
    v_fast = max(float(np.percentile(speeds, 90)), v_slow * 1.5)

    # Distance: patient ROM with floor
    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = ABS_D_MIN

    # Time: generous 2x margins on both ends
    t_min_cal = max(ABS_T_MIN, d_min_cal / (v_fast * 2.0))
    t_max_cal = min(ABS_T_MAX, (d_max_cal / v_slow) * 2.0)

    # Enforce minimum range widths so grid is never too narrow
    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15
    if t_max_cal - t_min_cal < 1.0:
        t_max_cal = t_min_cal + 1.0

    return (float(d_min_cal), float(min(d_max_cal, ABS_D_MAX)),
            float(t_min_cal), float(t_max_cal))


def expand_bounds_if_needed(d_min, d_max, t_min, t_max, observed_speeds):
    """
    Backup plan: if observed trial speeds suggest bounds are too tight, expand.
    Never shrinks bounds. Returns (d_min, d_max, t_min, t_max, changed).
    """
    ABS_T_MIN, ABS_T_MAX = 0.15, 15.0

    if len(observed_speeds) < 5:
        return d_min, d_max, t_min, t_max, False

    arr = np.array(observed_speeds[-50:])
    v_p5 = max(float(np.percentile(arr, 5)), 0.01)
    v_p95 = float(np.percentile(arr, 95))

    changed = False

    # Patient faster than expected → need shorter t_min
    new_t_min = max(ABS_T_MIN, d_min / (v_p95 * 2.5))
    if new_t_min < t_min * 0.7:
        t_min = new_t_min
        changed = True

    # Patient slower than expected → need longer t_max
    new_t_max = min(ABS_T_MAX, (d_max / v_p5) * 2.5)
    if new_t_max > t_max * 1.3:
        t_max = new_t_max
        changed = True

    return d_min, d_max, t_min, t_max, changed


def make_rom_state(*, n_directions: int = N_DIRECTIONS, d_min: float = D_MIN, d_max: float = D_MAX, n_bins: int = 12):
    edges = np.linspace(float(d_min), float(d_max), int(n_bins) + 1)
    shape = (int(n_directions), int(n_bins))
    return {
        "distance_edges": edges,
        "hit_counts": np.zeros(shape, dtype=float),
        "miss_counts": np.zeros(shape, dtype=float),
        "sum_dist_ratio_on_miss": np.zeros(shape, dtype=float),
    }


def rom_bin_index(d_sys: float, distance_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(distance_edges, float(d_sys), side="right") - 1)
    return int(np.clip(idx, 0, len(distance_edges) - 2))


def update_rom_state(rom_state: dict, *, direction: int, d_sys: float, hit: bool, dist_ratio: float):
    d_edges = np.asarray(rom_state["distance_edges"], dtype=float)
    b = rom_bin_index(d_sys, d_edges)
    direction = int(np.clip(direction, 0, rom_state["hit_counts"].shape[0] - 1))
    if hit:
        rom_state["hit_counts"][direction, b] += 1.0
    else:
        rom_state["miss_counts"][direction, b] += 1.0
        rom_state["sum_dist_ratio_on_miss"][direction, b] += float(np.clip(dist_ratio, 0.0, 1.0))


def rom_penalty_term(
    d_sys: float,
    direction: int,
    *,
    rom_state: dict | None,
    trial_index: int,
    warmup_trials: int = 40,
    w_rom: float = 0.45,
    tau_d: float = 0.05,
    boundary_threshold: float = 0.35,
    d_max: float = D_MAX,
):
    # Return additive objective penalty learned from miss patterns at larger distances.
    # No oracle max-reach is used; this is derived only from observed trial outcomes.
    if rom_state is None:
        return 0.0, 0.0, 0.0, float(d_max)

    hit_counts = np.asarray(rom_state["hit_counts"], dtype=float)
    miss_counts = np.asarray(rom_state["miss_counts"], dtype=float)
    miss_ratio_sum = np.asarray(rom_state["sum_dist_ratio_on_miss"], dtype=float)
    d_edges = np.asarray(rom_state["distance_edges"], dtype=float)
    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])

    direction = int(np.clip(direction, 0, hit_counts.shape[0] - 1))
    h = hit_counts[direction]
    m = miss_counts[direction]
    miss_ratio_sum_dir = miss_ratio_sum[direction]
    n = h + m

    miss_rate = (m + 1.0) / (n + 2.0)
    miss_severity = (m - miss_ratio_sum_dir + 1.0) / (m + 2.0)
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
    dir_conf = float(np.clip(np.sum(n) / 24.0, 0.0, 1.0))
    warmup = float(np.clip(float(trial_index) / max(float(warmup_trials), 1.0), 0.0, 1.0))
    tail_mask = d_centers >= float(d_sys)
    if np.any(tail_mask):
        tail_severity = float(np.mean(weighted_evidence[tail_mask]))
    else:
        tail_severity = float(np.mean(weighted_evidence))
    tail_severity = float(np.clip(tail_severity, 0.0, 1.0))

    risk = float(np.clip(dist_risk * dir_conf * max(tail_severity, 0.25), 0.0, 1.0))
    penalty = float(w_rom * warmup * risk)
    return penalty, risk, dir_conf, d_boundary


# ----------------------------
# OR Objective: score(d, t, direction)
# ----------------------------
def score_candidate(d, t, *,
                    v_hat, sigma_v,
                    direction: int,
                    p_star,
                    counts_5x5,
                    counts_dir,
                    dir_alpha: float,
                    dir_beta: float,
                    w_eff=1.0, w_var=0.20,
                    w_dir_eff=0.30,
                    w_dir_var=0.15,
                    p_min=0.10,
                    patient: PatientModel,
                    rom_state: dict | None = None,
                    trial_index: int = 0,
                    rom_warmup_trials: int = 40,
                    w_rom: float = 0.45,
                    rom_tau_d: float = 0.05,
                    rom_boundary_threshold: float = 0.35,
                    d_min: float = D_MIN, d_max: float = D_MAX,
                    t_min: float = T_MIN, t_max: float = T_MAX):
    p_dt = p_hit_from_speed(d, t, v_hat, sigma_v)

    # Direction hit probability from Beta posterior
    p_dir = float(dir_alpha) / (float(dir_alpha) + float(dir_beta))
    p_dir = float(np.clip(p_dir, 0.01, 0.99))

    # Combined p_hit: merge speed + direction via log-odds
    p_combined = combine_hit_probs_odds(p_dt, p_dir)

    if p_combined < p_min:
        return -1e9, p_combined, 0.0, 0.0, 0.0, float(d_max)

    # effort: keep combined predicted hit prob near p_star
    eff_raw = (p_combined - p_star)**2
    eff_normalized = 1.0 - eff_raw / (p_star**2)

    # variability bonus for (d,t): normalize to [0, 1]
    total = counts_5x5.sum()
    var_raw = rarity_bonus(counts_5x5, d, t, d_min, d_max, t_min, t_max)
    var_max = math.log(total + 25) if total > 0 else math.log(25)
    var_normalized = var_raw / (var_max + 1e-9)

    # direction exploitation: log-odds of per-direction hit rate
    log_odds_dir = math.log(p_dir / (1.0 - p_dir))
    # normalize from ~[-4.6, 4.6] to [0, 1]
    log_odds_normalized = float(np.clip((log_odds_dir + 4.6) / 9.2, 0.0, 1.0))

    # direction exploration: rarity bonus for under-sampled directions
    dir_total = counts_dir.sum()
    dir_var_raw = direction_rarity_bonus(counts_dir, direction)
    dir_var_max = math.log(dir_total + float(N_DIRECTIONS)) if dir_total > 0 else math.log(float(N_DIRECTIONS))
    dir_var_normalized = dir_var_raw / (dir_var_max + 1e-9)

    score = (w_eff * eff_normalized
             + w_var * var_normalized
             + w_dir_eff * log_odds_normalized
             + w_dir_var * dir_var_normalized)

    rom_penalty, rom_risk, rom_confidence, rom_boundary = rom_penalty_term(
        d,
        direction,
        rom_state=rom_state,
        trial_index=trial_index,
        warmup_trials=rom_warmup_trials,
        w_rom=w_rom,
        tau_d=rom_tau_d,
        boundary_threshold=rom_boundary_threshold,
        d_max=d_max,
    )
    score = score - rom_penalty

    return score, p_combined, rom_penalty, rom_risk, rom_confidence, rom_boundary


# ----------------------------
# Main loop
# ----------------------------
def run_sim(patient: PatientModel, n_trials=10000, seed=7, ema_alpha=0.20, calibration=True):
    rng = np.random.default_rng(seed)

    # --- Initial bounds (overridden by calibration if available) ---
    cur_d_min, cur_d_max = D_MIN, D_MAX
    cur_t_min, cur_t_max = T_MIN, T_MAX

    # objective targets
    p_star = 0.70
    p_min = 0.10

    counts_5x5 = np.zeros((5, 5), dtype=int)
    counts_dir = np.zeros(N_DIRECTIONS, dtype=int)

    d_prev, t_prev = None, None
    previous_hit = True

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)

        # Derive generous bounds from calibration data
        cur_d_min, cur_d_max, cur_t_min, cur_t_max = derive_bounds_from_calibration(
            calibration_result, patient)

    # --- Per-distance-bin speed model (uses calibration-derived bounds) ---
    N_SPEED_BINS = 3
    speed_bin_edges = np.linspace(cur_d_min, cur_d_max, N_SPEED_BINS + 1)
    sigma_v_floor = 0.02

    v_hat_init = 0.60
    sigma_v_init = 0.30
    v_hat_bins = np.full(N_SPEED_BINS, v_hat_init)
    sigma_v_bins = np.full(N_SPEED_BINS, sigma_v_init)
    m2_hat_bins = v_hat_bins**2 + sigma_v_bins**2

    def get_speed_bin(d):
        idx = int(np.searchsorted(speed_bin_edges, d, side='right') - 1)
        return int(np.clip(idx, 0, N_SPEED_BINS - 1))

    # Process calibration speed observations into distance bins
    if calibration_result:
        v_obs_per_bin = [[] for _ in range(N_SPEED_BINS)]
        for trial in calibration_result.get("trials", []):
            d_sys = float(trial.get("d_sys", 0.0))
            t_cap = float(trial.get("t_cap", 10.0))
            reached = bool(trial.get("reached", trial.get("hit", False)))
            if reached:
                t_pat_obs = float(trial.get("t_pat_obs", trial.get("t_pat", t_cap)))
                v_obs = d_sys / max(t_pat_obs, 1e-6)
            else:
                d_pat = float(trial.get("d_pat", float(trial.get("dist_ratio", 0.0)) * d_sys))
                v_obs = d_pat / max(t_cap, 1e-6)
            sb = get_speed_bin(d_sys)
            v_obs_per_bin[sb].append(v_obs)

        global_v_obs = []
        for sb in range(N_SPEED_BINS):
            if len(v_obs_per_bin[sb]) > 0:
                arr = np.array(v_obs_per_bin[sb])
                v_hat_bins[sb] = float(np.mean(arr))
                sigma_v_bins[sb] = float(max(np.std(arr), sigma_v_floor))
                m2_hat_bins[sb] = v_hat_bins[sb]**2 + sigma_v_bins[sb]**2
                global_v_obs.extend(v_obs_per_bin[sb])

        if len(global_v_obs) > 0:
            global_v = float(np.mean(global_v_obs))
            global_s = float(max(np.std(global_v_obs), sigma_v_floor))
            for sb in range(N_SPEED_BINS):
                if len(v_obs_per_bin[sb]) == 0:
                    v_hat_bins[sb] = global_v
                    sigma_v_bins[sb] = global_s
                    m2_hat_bins[sb] = global_v**2 + global_s**2

    # Build local candidate grids from calibration-derived bounds
    local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
    local_t_grid = np.round(np.arange(cur_t_min, cur_t_max + 1e-9, T_STEP), 4)

    # logs
    hist = {
        "d": [], "t": [], "v_req": [], "p_pred": [],
        "hit": [], "time_ratio": [], "dist_ratio": [],
        "v_hat": [], "sigma_v": [], "score": [], "direction": [],
        "rom_penalty": [], "rom_risk": [], "rom_confidence": [], "rom_boundary": []
    }
    rom_state = make_rom_state(d_min=cur_d_min, d_max=cur_d_max)
    observed_speeds = []

    for k in range(n_trials):

        # Backup plan: periodically check if bounds need expansion
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_d_min, new_d_max, new_t_min, new_t_max, changed = expand_bounds_if_needed(
                cur_d_min, cur_d_max, cur_t_min, cur_t_max, observed_speeds)
            if changed:
                cur_d_min, cur_d_max = new_d_min, new_d_max
                cur_t_min, cur_t_max = new_t_min, new_t_max
                local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
                local_t_grid = np.round(np.arange(cur_t_min, cur_t_max + 1e-9, T_STEP), 4)
                speed_bin_edges = np.linspace(cur_d_min, cur_d_max, N_SPEED_BINS + 1)

        # Joint optimization over (d, t, direction).
        best = None
        best_score = -1e18
        best_p = None
        best_direction = 2
        best_rom_penalty = 0.0
        best_rom_risk = 0.0
        best_rom_conf = 0.0
        best_rom_boundary = float(cur_d_max)

        for direction in range(N_DIRECTIONS):
            dir_a = float(patient.spatial_success_alpha[direction])
            dir_b = float(patient.spatial_success_beta[direction])
            for d in local_d_grid:
                sb = get_speed_bin(float(d))
                for t in local_t_grid:
                    sc, p_dt, rom_penalty, rom_risk, rom_conf, rom_boundary = score_candidate(
                        float(d), float(t),
                        v_hat=v_hat_bins[sb], sigma_v=sigma_v_bins[sb],
                        direction=direction,
                        dir_alpha=dir_a,
                        dir_beta=dir_b,
                        p_star=p_star,
                        counts_5x5=counts_5x5,
                        counts_dir=counts_dir,
                        w_eff=1.0, w_var=0.25,
                        w_dir_eff=0.30, w_dir_var=0.15,
                        p_min=p_min,
                        rom_state=rom_state,
                        trial_index=k,
                        rom_warmup_trials=40,
                        w_rom=0.45,
                        rom_tau_d=0.05,
                        rom_boundary_threshold=0.35,
                        patient=patient,
                        d_min=cur_d_min, d_max=cur_d_max,
                        t_min=cur_t_min, t_max=cur_t_max,
                    )

                    if sc > best_score:
                        best_score = sc
                        best = (float(d), float(t))
                        best_direction = direction
                        best_p = p_dt
                        best_rom_penalty = float(rom_penalty)
                        best_rom_risk = float(rom_risk)
                        best_rom_conf = float(rom_conf)
                        best_rom_boundary = float(rom_boundary)

        direction = best_direction
        d_sys, t_sys = best
        v_req = d_sys / max(t_sys, 1e-9)

        # update variability counts
        i, j = bin25(d_sys, t_sys, cur_d_min, cur_d_max, cur_t_min, cur_t_max)
        counts_5x5[i, j] += 1
        counts_dir[direction] += 1

        # run the trial using your simulator
        lvl = distance_level_from_patient_bins(patient, d_sys)
        outcome = patient.sample_trial(t_sys=t_sys, d_sys=d_sys,
                                       distance_level=lvl, previous_hit=previous_hit,
                                       direction_bin=direction)
        hit = bool(outcome["hit"])
        previous_hit = hit
        if hit:
            patient.spatial_success_alpha[direction] += 1.0
        else:
            patient.spatial_success_beta[direction] += 1.0
        update_rom_state(
            rom_state,
            direction=direction,
            d_sys=d_sys,
            hit=hit,
            dist_ratio=float(outcome["dist_ratio"]),
        )

        # online learning update: per-distance-bin speed tracking
        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])

        if hit:
            v_obs = d_pat / max(t_pat, 1e-6)
        else:
            v_obs = d_pat / max(t_sys, 1e-6)

        observed_speeds.append(v_obs)

        # Update the speed bin corresponding to this trial's distance
        sb = get_speed_bin(d_sys)
        alpha = float(np.clip(ema_alpha, 1e-6, 1.0))
        v_hat_bins[sb] = (1.0 - alpha) * v_hat_bins[sb] + alpha * v_obs
        m2_hat_bins[sb] = (1.0 - alpha) * m2_hat_bins[sb] + alpha * (v_obs ** 2)
        sigma_v_bins[sb] = float(np.sqrt(max(m2_hat_bins[sb] - v_hat_bins[sb]**2, sigma_v_floor**2)))

        # log
        hist["d"].append(d_sys)
        hist["t"].append(t_sys)
        hist["v_req"].append(v_req)
        hist["p_pred"].append(best_p)
        hist["hit"].append(int(hit))
        hist["time_ratio"].append(float(outcome["time_ratio"]))
        hist["dist_ratio"].append(float(outcome["dist_ratio"]))
        hist["v_hat"].append(float(v_hat_bins[sb]))
        hist["sigma_v"].append(float(sigma_v_bins[sb]))
        hist["score"].append(best_score)
        hist["direction"].append(direction)
        hist["rom_penalty"].append(best_rom_penalty)
        hist["rom_risk"].append(best_rom_risk)
        hist["rom_confidence"].append(best_rom_conf)
        hist["rom_boundary"].append(best_rom_boundary)

        d_prev, t_prev = d_sys, t_sys

    return hist, counts_5x5, patient
