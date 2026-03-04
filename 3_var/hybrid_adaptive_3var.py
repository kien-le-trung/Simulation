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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_MIN, D_MAX = 0.05, 1.0
T_MIN, T_MAX = 0.3, 7.0
D_STEP = 0.05
T_STEP = 0.25
N_DIRECTIONS = 5


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def level5(x, xmin, xmax):
    u = (x - xmin) / (xmax - xmin + 1e-12)
    if u < 0.2: return 0
    if u < 0.4: return 1
    if u < 0.6: return 2
    if u < 0.8: return 3
    return 4


def bin25(d, t, d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX):
    return (level5(d, d_min, d_max), level5(t, t_min, t_max))


def rarity_bonus(counts_5x5, d, t, d_min, d_max, t_min, t_max, eps=1e-9):
    i, j = bin25(d, t, d_min, d_max, t_min, t_max)
    total = counts_5x5.sum()
    freq = (counts_5x5[i, j] + 1.0) / (total + 25.0)
    return -math.log(freq + eps)


def direction_rarity_bonus(counts_dir, direction, eps=1e-9):
    total = counts_dir.sum()
    freq = (counts_dir[direction] + 1.0) / (total + float(len(counts_dir)))
    return -math.log(freq + eps)


def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    candidates = np.where(patient.d_means <= d_sys)[0]
    if len(candidates) == 0:
        return 0
    return int(candidates[-1])


def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_d(d, d_min, d_max):
    return (d - d_min) / (d_max - d_min + 1e-12)


def hard_time_feature(t, t_min, t_max):
    return 1.0 - (t - t_min) / (t_max - t_min + 1e-12)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
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

    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = ABS_D_MIN

    t_min_cal = max(ABS_T_MIN, d_min_cal / (v_fast * 2.0))
    t_max_cal = min(ABS_T_MAX, (d_max_cal / v_slow) * 2.0)

    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15
    if t_max_cal - t_min_cal < 1.0:
        t_max_cal = t_min_cal + 1.0

    return (float(d_min_cal), float(min(d_max_cal, ABS_D_MAX)),
            float(t_min_cal), float(t_max_cal))


def expand_bounds_if_needed(d_min, d_max, t_min, t_max, observed_speeds):
    ABS_T_MIN, ABS_T_MAX = 0.15, 15.0
    if len(observed_speeds) < 5:
        return d_min, d_max, t_min, t_max, False

    arr = np.array(observed_speeds[-50:])
    v_p5 = max(float(np.percentile(arr, 5)), 0.01)
    v_p95 = float(np.percentile(arr, 95))

    changed = False
    new_t_min = max(ABS_T_MIN, d_min / (v_p95 * 2.5))
    if new_t_min < t_min * 0.7:
        t_min = new_t_min
        changed = True
    new_t_max = min(ABS_T_MAX, (d_max / v_p5) * 2.5)
    if new_t_max > t_max * 1.3:
        t_max = new_t_max
        changed = True

    return d_min, d_max, t_min, t_max, changed


# ---------------------------------------------------------------------------
# ROM boundary learning (from OR)
# ---------------------------------------------------------------------------
def make_rom_state(*, n_directions=N_DIRECTIONS, d_min=D_MIN, d_max=D_MAX, n_bins=12):
    edges = np.linspace(float(d_min), float(d_max), int(n_bins) + 1)
    shape = (int(n_directions), int(n_bins))
    return {
        "distance_edges": edges,
        "hit_counts": np.zeros(shape, dtype=float),
        "miss_counts": np.zeros(shape, dtype=float),
        "sum_dist_ratio_on_miss": np.zeros(shape, dtype=float),
    }


def rom_bin_index(d_sys, distance_edges):
    idx = int(np.searchsorted(distance_edges, float(d_sys), side="right") - 1)
    return int(np.clip(idx, 0, len(distance_edges) - 2))


def update_rom_state(rom_state, *, direction, d_sys, hit, dist_ratio):
    d_edges = np.asarray(rom_state["distance_edges"], dtype=float)
    b = rom_bin_index(d_sys, d_edges)
    direction = int(np.clip(direction, 0, rom_state["hit_counts"].shape[0] - 1))
    if hit:
        rom_state["hit_counts"][direction, b] += 1.0
    else:
        rom_state["miss_counts"][direction, b] += 1.0
        rom_state["sum_dist_ratio_on_miss"][direction, b] += float(np.clip(dist_ratio, 0.0, 1.0))


def rom_penalty_term(d_sys, direction, *, rom_state, trial_index,
                     warmup_trials=40, w_rom=0.45, tau_d=0.05,
                     boundary_threshold=0.35, d_max=D_MAX):
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


# ---------------------------------------------------------------------------
# Direction-blind logistic regression model (d/t only, direction via Beta posterior)
# ---------------------------------------------------------------------------
class DifficultyIndexLogReg:
    def __init__(self, rng, beta0=0.0, beta_d=-1.0, beta_t=-1.0,
                 lr_beta=0.25, p_star=0.70,
                 d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX):
        self.rng = rng
        self.beta0 = float(beta0)
        self.beta_d = float(beta_d)
        self.beta_t = float(beta_t)

        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        self.lr_beta = float(lr_beta)
        self.p_star = float(p_star)
        self._seen = set()

    def predict_p(self, d, t):
        if len(self._seen) < 2:
            return 0.5
        dn = normalize_d(d, self.d_min, self.d_max)
        ht = hard_time_feature(t, self.t_min, self.t_max)
        z = self.beta0 + self.beta_d * dn + self.beta_t * ht
        return float(sigmoid(z))

    def update(self, hit, d_exec, t_exec):
        y = 1.0 if hit else 0.0
        self._seen.add(int(y))

        dn = normalize_d(d_exec, self.d_min, self.d_max)
        ht = hard_time_feature(t_exec, self.t_min, self.t_max)
        z = self.beta0 + self.beta_d * dn + self.beta_t * ht
        p_exec = sigmoid(z)

        g = (p_exec - y)
        self.beta0 -= self.lr_beta * g
        self.beta_d -= self.lr_beta * g * dn
        self.beta_t -= self.lr_beta * g * ht

        self.beta0 = float(np.clip(self.beta0, -12.0, 12.0))
        self.beta_d = float(np.clip(self.beta_d, -8.0, 8.0))
        self.beta_t = float(np.clip(self.beta_t, -8.0, 8.0))

        return float(p_exec)


def combine_hit_probs_odds(p_dt, p_dir, eps=1e-9):
    p_dt = float(np.clip(p_dt, eps, 1.0 - eps))
    p_dir = float(np.clip(p_dir, eps, 1.0 - eps))
    odds_dt = p_dt / (1.0 - p_dt)
    odds_dir = p_dir / (1.0 - p_dir)
    odds = odds_dt * odds_dir
    return odds / (1.0 + odds)


# ---------------------------------------------------------------------------
# Hybrid scoring function (Logistic prediction + OR-style composite objective)
# ---------------------------------------------------------------------------
def score_candidate(d, t, direction, *, model, p_star, counts_5x5, counts_dir,
                    dir_alpha, dir_beta,
                    rom_state, trial_index,
                    w_eff=1.0, w_var=0.20, w_dir_eff=0.30, w_dir_var=0.15,
                    p_min=0.10,
                    d_min, d_max, t_min, t_max):
    # Direction-blind logistic prediction for d/t difficulty
    p_dt = model.predict_p(d, t)

    # Direction hit probability from Beta posterior
    p_dir = float(dir_alpha) / (float(dir_alpha) + float(dir_beta))
    p_dir = float(np.clip(p_dir, 0.01, 0.99))

    # Combined p_hit: merge d/t difficulty + direction via log-odds (like OR)
    p_pred = combine_hit_probs_odds(p_dt, p_dir)

    if p_pred < p_min:
        return -1e9, p_pred, 0.0, 0.0, 0.0, float(d_max)

    # 1. Efficiency: keep predicted hit rate near p_star
    eff_raw = (p_pred - p_star) ** 2
    eff_normalized = 1.0 - eff_raw / (p_star ** 2)

    # 2. d/t exploration: rarity bonus (from OR)
    total = counts_5x5.sum()
    var_raw = rarity_bonus(counts_5x5, d, t, d_min, d_max, t_min, t_max)
    var_max = math.log(total + 25) if total > 0 else math.log(25)
    var_normalized = var_raw / (var_max + 1e-9)

    # 3. Direction exploitation: Beta posterior (from OR)
    log_odds_dir = math.log(p_dir / (1.0 - p_dir))
    log_odds_normalized = float(np.clip((log_odds_dir + 4.6) / 9.2, 0.0, 1.0))

    # 4. Direction exploration: rarity bonus (from OR)
    dir_total = counts_dir.sum()
    dir_var_raw = direction_rarity_bonus(counts_dir, direction)
    dir_var_max = math.log(dir_total + float(N_DIRECTIONS)) if dir_total > 0 else math.log(float(N_DIRECTIONS))
    dir_var_normalized = dir_var_raw / (dir_var_max + 1e-9)

    score = (w_eff * eff_normalized
             + w_var * var_normalized
             + w_dir_eff * log_odds_normalized
             + w_dir_var * dir_var_normalized)

    # 5. ROM penalty (from OR)
    rom_penalty, rom_risk, rom_conf, rom_boundary = rom_penalty_term(
        d, direction,
        rom_state=rom_state,
        trial_index=trial_index,
        warmup_trials=40,
        w_rom=0.45,
        tau_d=0.05,
        boundary_threshold=0.35,
        d_max=d_max,
    )
    score = score - rom_penalty

    return score, p_pred, rom_penalty, rom_risk, rom_conf, rom_boundary


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_sim(patient: PatientModel, n_trials=500, seed=7, calibration=True):
    rng = np.random.default_rng(seed)

    p_star = 0.70

    cur_d_min, cur_d_max = D_MIN, D_MAX
    cur_t_min, cur_t_max = T_MIN, T_MAX

    # --- Calibration ---
    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        cur_d_min, cur_d_max, cur_t_min, cur_t_max = derive_bounds_from_calibration(
            calibration_result, patient)

    # --- Initialize direction-blind logistic model ---
    model = DifficultyIndexLogReg(
        rng=rng,
        beta0=0.0, beta_d=-1.0, beta_t=-1.0,
        lr_beta=0.30,
        p_star=p_star,
        d_min=cur_d_min, d_max=cur_d_max,
        t_min=cur_t_min, t_max=cur_t_max,
    )

    # Seed model from calibration trials
    if calibration_result:
        for trial in calibration_result.get("trials", []):
            d_exec = float(trial.get("d_sys", cur_d_min))
            t_exec = float(np.clip(trial.get("t_cap", cur_t_max), cur_t_min, cur_t_max))
            hit = bool(trial.get("hit", False))
            model.update(hit, d_exec, t_exec)

    # --- Build candidate grid ---
    local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
    local_t_grid = np.round(np.arange(cur_t_min, cur_t_max + 1e-9, T_STEP), 4)

    # --- Tracking state ---
    counts_5x5 = np.zeros((5, 5), dtype=int)
    counts_dir = np.zeros(N_DIRECTIONS, dtype=int)
    rom_state = make_rom_state(d_min=cur_d_min, d_max=cur_d_max)
    observed_speeds = []

    hist = {
        "d": [], "t": [], "direction": [],
        "p_pred": [], "score": [],
        "hit": [], "time_ratio": [], "dist_ratio": [],
        "t_pat": [], "d_pat": [],
        "rom_penalty": [], "rom_risk": [], "rom_confidence": [], "rom_boundary": [],
    }

    previous_hit = True

    for k in range(n_trials):
        # Backup plan: check bounds expansion every 50 trials
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_d_min, new_d_max, new_t_min, new_t_max, changed = expand_bounds_if_needed(
                cur_d_min, cur_d_max, cur_t_min, cur_t_max, observed_speeds)
            if changed:
                cur_d_min, cur_d_max = new_d_min, new_d_max
                cur_t_min, cur_t_max = new_t_min, new_t_max
                model.d_min, model.d_max = cur_d_min, cur_d_max
                model.t_min, model.t_max = cur_t_min, cur_t_max
                local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
                local_t_grid = np.round(np.arange(cur_t_min, cur_t_max + 1e-9, T_STEP), 4)

        # --- Joint optimization over (d, t, direction) ---
        best_score = -1e18
        best_d, best_t, best_direction = float(local_d_grid[0]), float(local_t_grid[0]), 2
        best_p_pred = 0.5
        best_rom_penalty = 0.0
        best_rom_risk = 0.0
        best_rom_conf = 0.0
        best_rom_boundary = float(cur_d_max)

        for direction in range(N_DIRECTIONS):
            dir_a = float(patient.spatial_success_alpha[direction])
            dir_b = float(patient.spatial_success_beta[direction])
            for d in local_d_grid:
                for t in local_t_grid:
                    sc, p_pred, rom_pen, rom_risk, rom_conf, rom_boundary = score_candidate(
                        float(d), float(t), direction,
                        model=model,
                        p_star=p_star,
                        counts_5x5=counts_5x5,
                        counts_dir=counts_dir,
                        dir_alpha=dir_a,
                        dir_beta=dir_b,
                        rom_state=rom_state,
                        trial_index=k,
                        d_min=cur_d_min, d_max=cur_d_max,
                        t_min=cur_t_min, t_max=cur_t_max,
                    )
                    if sc > best_score:
                        best_score = sc
                        best_d, best_t = float(d), float(t)
                        best_direction = direction
                        best_p_pred = p_pred
                        best_rom_penalty = float(rom_pen)
                        best_rom_risk = float(rom_risk)
                        best_rom_conf = float(rom_conf)
                        best_rom_boundary = float(rom_boundary)

        d_sys, t_sys = best_d, best_t
        direction = best_direction

        # Update counts
        i, j = bin25(d_sys, t_sys, cur_d_min, cur_d_max, cur_t_min, cur_t_max)
        counts_5x5[i, j] += 1
        counts_dir[direction] += 1

        # Execute trial
        lvl = distance_level_from_patient_bins(patient, d_sys)
        outcome = patient.sample_trial(
            t_sys=t_sys, d_sys=d_sys,
            distance_level=lvl, previous_hit=previous_hit,
            direction_bin=direction,
        )
        hit = bool(outcome["hit"])
        previous_hit = hit

        # Update patient Beta posteriors
        if hit:
            patient.spatial_success_alpha[direction] += 1.0
        else:
            patient.spatial_success_beta[direction] += 1.0

        # Update ROM state
        update_rom_state(
            rom_state,
            direction=direction,
            d_sys=d_sys,
            hit=hit,
            dist_ratio=float(outcome["dist_ratio"]),
        )

        # Track observed speed for bounds expansion
        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])
        if hit and t_pat > 0.01:
            observed_speeds.append(d_pat / t_pat)
        elif t_sys > 0.01:
            observed_speeds.append(d_pat / t_sys)

        # Update logistic model via SGD (direction-blind)
        model.update(hit, d_sys, t_sys)

        # Log
        hist["d"].append(d_sys)
        hist["t"].append(t_sys)
        hist["direction"].append(direction)
        hist["p_pred"].append(best_p_pred)
        hist["score"].append(best_score)
        hist["hit"].append(int(hit))
        hist["time_ratio"].append(float(outcome["time_ratio"]))
        hist["dist_ratio"].append(float(outcome["dist_ratio"]))
        hist["t_pat"].append(float(outcome["t_pat"]))
        hist["d_pat"].append(float(outcome["d_pat"]))
        hist["rom_penalty"].append(best_rom_penalty)
        hist["rom_risk"].append(best_rom_risk)
        hist["rom_confidence"].append(best_rom_conf)
        hist["rom_boundary"].append(best_rom_boundary)

    return hist, counts_5x5, patient
