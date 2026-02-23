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
D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0

# resolution (tune for speed vs granularity)
D_STEP = 0.05   # meters
T_STEP = 0.25   # seconds

d_grid = np.round(np.arange(D_MIN, D_MAX + 1e-9, D_STEP), 4)
t_grid = np.round(np.arange(T_MIN, T_MAX + 1e-9, T_STEP), 4)
N_DIRECTIONS = 9
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

def bin25(d, t):
    return (level5(d, D_MIN, D_MAX), level5(t, T_MIN, T_MAX))  # (dist_level, time_level)

def rarity_bonus(counts_5x5, d, t, eps=1e-9):
    """Bonus for under-sampled bins (approx -log frequency)."""
    i, j = bin25(d, t)
    total = counts_5x5.sum()
    # Laplace smoothing
    freq = (counts_5x5[i, j] + 1.0) / (total + 25.0)
    return -math.log(freq + eps)

def smooth_penalty(d, t, d_prev, t_prev):
    """Quadratic penalty for jumping in normalized space."""
    if d_prev is None or t_prev is None:
        return 0.0
    dn = (d - d_prev) / (D_MAX - D_MIN + 1e-12)
    tn = (t - t_prev) / (T_MAX - T_MIN + 1e-12)
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


# ----------------------------
# Map continuous d to your PatientModel distance_level (0..7)
# Uses the rule in your comment: closest bucket with d_mean <= d_sys
# ----------------------------
def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    candidates = np.where(patient.d_means <= d_sys)[0]
    if len(candidates) == 0:
        return 0
    return int(candidates[-1])


def nearest_level_index(levels: np.ndarray, value: float) -> int:
    levels = np.asarray(levels, dtype=float)
    if levels.size == 0:
        return 0
    return int(np.argmin(np.abs(levels - value)))


def dt_bin_from_patient_levels(patient: PatientModel, d_sys: float, t_sys: float):
    d_idx = nearest_level_index(patient.d_levels, d_sys)
    t_idx = nearest_level_index(patient.t_levels, t_sys)
    return d_idx, t_idx


def wr_from_bin_stats(
    d_idx: int,
    t_idx: int,
    *,
    hit_counts: np.ndarray,
    miss_counts: np.ndarray,
    sum_t_ratio_hit: np.ndarray,
    sum_d_ratio_miss: np.ndarray,
    alpha_hit: float = 1.0,
    beta_hit: float = 1.0,
    t_ratio_prior: float = 0.5,
    d_ratio_prior: float = 0.5,
):
    n_hit = float(hit_counts[d_idx, t_idx])
    n_miss = float(miss_counts[d_idx, t_idx])

    p_hit = (n_hit + alpha_hit) / (n_hit + n_miss + alpha_hit + beta_hit)
    mu_t_hit = (sum_t_ratio_hit[d_idx, t_idx] + t_ratio_prior) / (n_hit + 1.0)
    mu_d_miss = (sum_d_ratio_miss[d_idx, t_idx] + d_ratio_prior) / (n_miss + 1.0)
    mu_t_hit = float(np.clip(mu_t_hit, 0.0, 1.0))
    mu_d_miss = float(np.clip(mu_d_miss, 0.0, 1.0))

    # Expected deficit:
    #   W = p_hit * E[t_ratio | hit] + (1-p_hit) * E[1-d_ratio | miss]
    wr = p_hit * mu_t_hit + (1.0 - p_hit) * (1.0 - mu_d_miss)
    return float(np.clip(wr, 0.0, 1.0))


# ----------------------------
# Main loop
# ----------------------------
def run_sim(patient: PatientModel, n_trials=10000, seed=7):
    rng = np.random.default_rng(seed)

    # Online estimates (speed model)
    v_patient = [] 
    v_hat = 0.25          # initial guess (m/s) until we have data
    sigma_v = 0.1        # initial uncertainty in effective speed

    # objective targets
    p_star = 0.70         # desired hit probability
    p_min = 0.10          # safety: don't choose near-impossible tasks
    p_gate_low = 0.5
    p_gate_high = 0.8

    # Stage-2 sampling schedule:
    # start near-uniform random, then transition toward Wr-biased sampling.
    bias_warmup_trials = 250
    softmax_tau = 6.0

    counts_5x5 = np.zeros((5, 5), dtype=int)
    n_d_bins = len(np.asarray(patient.d_levels))
    n_t_bins = len(np.asarray(patient.t_levels))
    hit_counts = np.zeros((n_d_bins, n_t_bins), dtype=int)
    miss_counts = np.zeros((n_d_bins, n_t_bins), dtype=int)
    sum_t_ratio_hit = np.zeros((n_d_bins, n_t_bins), dtype=float)
    sum_d_ratio_miss = np.zeros((n_d_bins, n_t_bins), dtype=float)

    d_prev, t_prev = None, None
    previous_hit = True

    # logs
    hist = {
        "d": [], "t": [], "v_req": [], "p_pred": [],
        "hit": [], "time_ratio": [], "dist_ratio": [],
        "v_hat": [], "sigma_v": [], "score": [], "direction": [],
        "w_rom": [], "w_time": [], "reward": [], "reward_baseline": [],
        "wr": [], "bias_lambda": [], "dt_bin_d": [], "dt_bin_t": []
    }

    for k in range(n_trials):
        # Thompson sample direction and pick closest to 0.7
        dir_samples = rng.beta(patient.spatial_success_alpha,
                               patient.spatial_success_beta)
        direction = int(np.argmin(np.abs(dir_samples - 0.7)))
        # randomly choose a direction
        # direction = rng.integers(0, N_DIRECTIONS)
        p_dir = float(dir_samples[direction])

        # ----------------------------
        # Stage 1: gateway by hit probability only
        # ----------------------------
        gateway_candidates = []
        feasible_candidates = []
        for (d, t, cand_dir) in CANDIDATES:
            if int(cand_dir) != direction:
                continue
            p_dt = p_hit_from_speed(d, t, v_hat, sigma_v)
            p = combine_hit_probs_odds(p_dt, p_dir)
            if p < p_min:
                continue
            cand = (float(d), float(t), float(p))
            feasible_candidates.append(cand)
            if p_gate_low < p < p_gate_high:
                gateway_candidates.append(cand)

        if len(gateway_candidates) > 0:
            candidate_pool = gateway_candidates
        else:
            candidate_pool = feasible_candidates

        if len(candidate_pool) == 0:
            # Safety fallback: choose closest to p_min among direction candidates.
            # This should be rare and only occurs if model estimate is overly pessimistic.
            all_dir = [(float(d), float(t)) for (d, t, cand_dir) in CANDIDATES if int(cand_dir) == direction]
            d_sys, t_sys = all_dir[int(rng.integers(0, len(all_dir)))]
            best_p = p_min
            best_score = 0.0
            chosen_wr = 0.0
            chosen_bin = dt_bin_from_patient_levels(patient, d_sys, t_sys)
            bias_lambda = 0.0
        else:
            # ----------------------------
            # Stage 2: Wr scoring by (d,t)-bin and biased sampling
            # ----------------------------
            wr_vals = []
            for (d, t, _) in candidate_pool:
                d_idx, t_idx = dt_bin_from_patient_levels(patient, d, t)
                wr_vals.append(
                    wr_from_bin_stats(
                        d_idx,
                        t_idx,
                        hit_counts=hit_counts,
                        miss_counts=miss_counts,
                        sum_t_ratio_hit=sum_t_ratio_hit,
                        sum_d_ratio_miss=sum_d_ratio_miss,
                    )
                )
            wr_arr = np.asarray(wr_vals, dtype=float)
            n_pool = len(candidate_pool)

            # 0 -> random uniform, 1 -> fully Wr-biased
            bias_lambda = float(min(1.0, k / max(1.0, float(bias_warmup_trials))))
            wr_logits = softmax_tau * (wr_arr - np.max(wr_arr))
            wr_probs = np.exp(wr_logits)
            wr_probs /= np.sum(wr_probs)
            pick_probs = (1.0 - bias_lambda) * (np.ones(n_pool) / n_pool) + bias_lambda * wr_probs
            pick_probs = pick_probs / np.sum(pick_probs)

            pick_idx = int(rng.choice(n_pool, p=pick_probs))
            d_sys, t_sys, best_p = candidate_pool[pick_idx]
            chosen_wr = float(wr_arr[pick_idx])
            chosen_bin = dt_bin_from_patient_levels(patient, d_sys, t_sys)
            best_score = chosen_wr

        v_req = d_sys / max(t_sys, 1e-9)

        # update variability counts using the chosen bin
        i, j = bin25(d_sys, t_sys)
        counts_5x5[i, j] += 1

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

        # online learning update: separate speed tracking for hits vs misses
        # calculate v_obs = d_patient / t_patient from outcome
        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])

        if hit:
            # On hits: d_pat = d_sys (always reaches target), t_pat = actual time
            v_obs_hit = d_pat / max(t_pat, 1e-6)
            v_patient.append(v_obs_hit)
        else:
            # On misses: use distance achieved over time allowed (not t_pat)
            v_obs_miss = d_pat / max(t_sys, 1e-6)
            v_patient.append(v_obs_miss)

        
        v_hat = float(np.mean(v_patient))
        if len(v_patient) >= 2:
            sigma_v = float(np.std(v_patient, ddof=1))

        # Update Wr stats in the selected (d,t) bin.
        d_idx, t_idx = chosen_bin
        if hit:
            hit_counts[d_idx, t_idx] += 1
            sum_t_ratio_hit[d_idx, t_idx] += float(outcome["time_ratio"])
        else:
            miss_counts[d_idx, t_idx] += 1
            sum_d_ratio_miss[d_idx, t_idx] += float(outcome["dist_ratio"])

        # Legacy fields retained for compatibility with existing plotting code.
        w_rom = 0.0
        w_time = 0.0
        reward = chosen_wr
        reward_baseline = chosen_wr

        # log
        hist["d"].append(d_sys)
        hist["t"].append(t_sys)
        hist["v_req"].append(v_req)
        hist["p_pred"].append(best_p)
        hist["hit"].append(int(hit))
        hist["time_ratio"].append(float(outcome["time_ratio"]))
        hist["dist_ratio"].append(float(outcome["dist_ratio"]))
        hist["v_hat"].append(v_hat)
        hist["sigma_v"].append(sigma_v)
        hist["score"].append(best_score)
        hist["direction"].append(direction)
        hist["w_rom"].append(w_rom)
        hist["w_time"].append(w_time)
        hist["reward"].append(reward)
        hist["reward_baseline"].append(reward_baseline)
        hist["wr"].append(chosen_wr)
        hist["bias_lambda"].append(bias_lambda)
        hist["dt_bin_d"].append(int(d_idx))
        hist["dt_bin_t"].append(int(t_idx))

        d_prev, t_prev = d_sys, t_sys

    return hist, counts_5x5, patient
