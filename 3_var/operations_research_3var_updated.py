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


# ----------------------------
# OR Objective: score(d,t)
# ----------------------------
def score_candidate(d, t, *,
                    v_hat, sigma_v,
                    p_dir,
                    p_star,
                    counts_5x5,
                    d_prev, t_prev,
                    w_eff=1.0, w_var=0.25, w_smooth=0.40,
                    p_min=0.10,
                    patient: PatientModel):
    # effort: keep predicted hit prob near p_star (hard but doable)
    p_dt = p_hit_from_speed(d, t, v_hat, sigma_v)
    p = combine_hit_probs_odds(p_dt, p_dir)
    if p < p_min:
        return -1e9, p  # hard safety filter

    # effort: normalize squared error to [0, 1]
    # max error is (0 - p_star)^2 = p_star^2
    eff_raw = (p - p_star)**2
    eff_normalized = 1.0 - eff_raw / (p_star**2)  # 1 = perfect match, 0 = worst

    # variability bonus: normalize to [0, 1] using theoretical bounds
    # With Laplace smoothing: freq = (count + 1) / (total + 25)
    # Theoretical max: -log(1/(total+25)) = log(total+25)
    # Theoretical min: -log(1) = 0
    total = counts_5x5.sum()
    var_raw = rarity_bonus(counts_5x5, d, t)
    var_max = math.log(total + 25) if total > 0 else math.log(25)
    var_normalized = var_raw / (var_max + 1e-9)  # 1 = maximally rare bin

    score = w_eff * eff_normalized + w_var * var_normalized

    """Interestingly, adding the rom and speed score does not really change the overall behavior much, but it gives prioritize to  higher d and t"""

    return score, p


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

    counts_5x5 = np.zeros((5, 5), dtype=int)

    d_prev, t_prev = None, None
    previous_hit = True

    # logs
    hist = {
        "d": [], "t": [], "v_req": [], "p_pred": [],
        "hit": [], "time_ratio": [], "dist_ratio": [],
        "v_hat": [], "sigma_v": [], "score": [], "direction": []
    }

    for k in range(n_trials):
        # Thompson sample direction and pick closest to 0.7
        dir_samples = rng.beta(patient.spatial_success_alpha,
                               patient.spatial_success_beta)
        direction = int(np.argmin(np.abs(dir_samples - 0.7)))
        # randomly choose a direction
        # direction = rng.integers(0, N_DIRECTIONS)
        p_dir = float(dir_samples[direction])

        # choose best candidate on lattice (for chosen direction)
        best = None
        best_score = -1e18
        best_p = None

        for (d, t, cand_dir) in CANDIDATES:
            if int(cand_dir) != direction:
                continue
            sc, p = score_candidate(
                d, t,
                v_hat=v_hat, sigma_v=sigma_v,
                p_dir=p_dir,
                p_star=p_star,
                counts_5x5=counts_5x5,
                d_prev=d_prev, t_prev=t_prev,
                w_eff=1.0, w_var=0.40, w_smooth=0.40,
                p_min=p_min,
                patient=patient
            )
            if sc > best_score:
                best_score = sc
                best = (float(d), float(t))
                best_p = p

        d_sys, t_sys = best
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

        d_prev, t_prev = d_sys, t_sys

    return hist, counts_5x5, patient
