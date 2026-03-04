from __future__ import annotations
# rl_logreg_difficulty_index.py
# Direct logistic regression on normalized distance/time with online SGD.
#
# Requires: numpy, matplotlib
# Uses your patient simulator: patient_simulation_v4.py

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

# -----------------------------
# Candidate grid for (d, t)
# -----------------------------
D_MIN, D_MAX = 0.10, 1.0
T_MIN, T_MAX = 0.3, 7.0

D_STEP = 0.05
T_STEP = 0.5

D_GRID = np.round(np.arange(D_MIN, D_MAX + 1e-9, D_STEP), 4)
T_GRID = np.round(np.arange(T_MIN, T_MAX + 1e-9, T_STEP), 4)
CANDIDATES = [(float(d), float(t)) for d in D_GRID for t in T_GRID]


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


def bin25(d, t, d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX):
    return (level5(d, d_min, d_max), level5(t, t_min, t_max))


def pick_with_tiebreak(candidates, score_fn, rng, maximize=False, atol=1e-12):
    scores = np.array([float(score_fn(c)) for c in candidates], dtype=float)
    best_score = float(np.max(scores) if maximize else np.min(scores))
    tie_idx = np.where(np.isclose(scores, best_score, atol=atol, rtol=0.0))[0]
    chosen = int(rng.choice(tie_idx))
    return candidates[chosen]


# -----------------------------
# Map continuous d_sys to patient distance_level
# -----------------------------
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

    new_t_min = max(ABS_T_MIN, d_min / (v_p95 * 2.5))
    if new_t_min < t_min * 0.7:
        t_min = new_t_min
        changed = True

    new_t_max = min(ABS_T_MAX, (d_max / v_p5) * 2.5)
    if new_t_max > t_max * 1.3:
        t_max = new_t_max
        changed = True

    return d_min, d_max, t_min, t_max, changed


# -----------------------------
# Jitter (harder/easier execution)
# -----------------------------
def apply_jitter(d, t, rng, p_jitter=0.7, d_sigma=0.06, t_sigma=0.10,
                 d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX):
    d_sys, t_sys = d, t
    if rng.random() < p_jitter:
        if rng.random() < 0.5:
            eps = rng.normal(0.0, d_sigma)
            d_sys = float(np.clip(d * (1.0 + eps), d_min, d_max))
        else:
            eps = rng.normal(0.0, t_sigma)
            t_sys = float(np.clip(t * (1.0 + eps), t_min, t_max))
    return d_sys, t_sys


# -----------------------------
# Direct logistic model
# -----------------------------
def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_d(d, d_min=D_MIN, d_max=D_MAX):
    return (d - d_min) / (d_max - d_min + 1e-12)


def hard_time_feature(t, t_min=T_MIN, t_max=T_MAX):
    # bigger means harder: shortest time => 1, longest time => 0
    return 1.0 - (t - t_min) / (t_max - t_min + 1e-12)


class LogisticDTReg:
    """
    Direct logistic regression over distance/time:
      p_hat = sigmoid(beta0 + beta_d * d_norm + beta_t * hard_time)
    """

    def __init__(
        self,
        rng,
        beta0=0.0,
        beta_d=-1.0,
        beta_t=-1.0,
        lr_beta=0.25,
        p_star=0.70,
        d_min=D_MIN, d_max=D_MAX, t_min=T_MIN, t_max=T_MAX,
    ):
        self.rng = rng
        self.beta0 = float(beta0)
        self.beta_d = float(beta_d)
        self.beta_t = float(beta_t)

        self.lr_beta = float(lr_beta)

        self.p_star = float(p_star)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        self._seen = set()  # track if we've seen both classes for stable behavior

    def predict_p(self, d, t):
        # before seeing both classes, be maximally uncertain
        if len(self._seen) < 2:
            return 0.5
        dn = normalize_d(d, self.d_min, self.d_max)
        ht = hard_time_feature(t, self.t_min, self.t_max)
        return float(sigmoid(self.beta0 + self.beta_d * dn + self.beta_t * ht))

    def uncertainty(self, d, t):
        p = self.predict_p(d, t)
        return 1.0 - 2.0 * abs(p - 0.5)  # in [0,1]

    def update(self, hit, d_exec, t_exec):
        """
        Supervised update for beta from executed (d_exec, t_exec) and hit label.
        """
        y = 1.0 if hit else 0.0
        self._seen.add(int(y))

        dn = normalize_d(d_exec, self.d_min, self.d_max)
        ht = hard_time_feature(t_exec, self.t_min, self.t_max)
        p_exec = sigmoid(self.beta0 + self.beta_d * dn + self.beta_t * ht)

        # gradient of log-loss: (p - y)
        g = p_exec - y
        self.beta0 -= self.lr_beta * g
        self.beta_d -= self.lr_beta * g * dn
        self.beta_t -= self.lr_beta * g * ht

        # stabilize extreme drift
        self.beta0 = float(np.clip(self.beta0, -12.0, 12.0))
        self.beta_d = float(np.clip(self.beta_d, -8.0, 8.0))
        self.beta_t = float(np.clip(self.beta_t, -8.0, 8.0))
        return float(p_exec)


# -----------------------------
# Controller run
# -----------------------------
def run_controller(
    patient: PatientModel,
    n_trials=200,
    seed=7,
    p_star=0.70,
    p_tol=0.05,
    explore_prob=0.10,  # active-learning exploration near p=0.5
    p_jitter=0.30,
    calibration=True,
):
    rng = np.random.default_rng(seed)
    cur_d_min, cur_d_max = D_MIN, D_MAX
    cur_t_min, cur_t_max = T_MIN, T_MAX

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        cur_d_min, cur_d_max, cur_t_min, cur_t_max = derive_bounds_from_calibration(
            calibration_result, patient)

    model = LogisticDTReg(
        rng=rng,
        beta0=0.0,
        beta_d=-1.0,
        beta_t=-1.0,
        lr_beta=0.30,
        p_star=p_star,
        d_min=cur_d_min, d_max=cur_d_max,
        t_min=cur_t_min, t_max=cur_t_max,
    )

    counts_5x5 = np.zeros((5, 5), dtype=int)

    hist = {
        "a_i": [],
        "a_j": [],
        "d": [],
        "t": [],
        "d_sys": [],
        "t_sys": [],
        "p_pred": [],
        "uncert": [],
        "hit": [],
        "rolling_hit": [],
        "beta0": [],
        "beta_d": [],
        "beta_t": [],
    }

    prev_hit = True
    rolling_window = 20
    hit_list = []

    if calibration_result:
        for trial in calibration_result.get("trials", []):
            d_exec = float(trial.get("d_sys", cur_d_min))
            t_exec = float(np.clip(trial.get("t_cap", cur_t_max), cur_t_min, cur_t_max))
            hit = bool(trial.get("hit", False))
            model.update(hit, d_exec, t_exec)

    local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
    local_t_grid = np.round(np.arange(cur_t_min, cur_t_max + 1e-9, T_STEP), 4)
    local_candidates = [(float(d), float(t)) for d in local_d_grid for t in local_t_grid]

    observed_speeds = []

    for k in range(n_trials):
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
                local_candidates = [(float(d), float(t)) for d in local_d_grid for t in local_t_grid]

        # score full candidate grid, then keep only near-target p(hit)
        scored = []
        for d, t in local_candidates:
            p_pred = model.predict_p(d, t)
            uncert = model.uncertainty(d, t)
            scored.append((d, t, p_pred, uncert))

        near_target = [x for x in scored if abs(x[2] - p_star) <= p_tol]
        pool = near_target if len(near_target) > 0 else scored

        # choose candidate with exploration probability
        if rng.random() < explore_prob:
            # exploration: pick highest-uncertainty candidate in pool
            d, t, p_pred_intended, uncert_val = pick_with_tiebreak(
                pool, score_fn=lambda x: x[3], rng=rng, maximize=True
            )
        else:
            # exploitation: best p(hit) match to target in pool
            d, t, p_pred_intended, uncert_val = pick_with_tiebreak(
                pool, score_fn=lambda x: abs(x[2] - p_star), rng=rng, maximize=False
            )

        i, j = bin25(d, t, cur_d_min, cur_d_max, cur_t_min, cur_t_max)

        # apply jitter to executed system values
        d_sys, t_sys = apply_jitter(
            d, t, rng, p_jitter=p_jitter,
            d_min=cur_d_min, d_max=cur_d_max,
            t_min=cur_t_min, t_max=cur_t_max,
        )

        # run patient trial
        lvl = distance_level_from_patient_bins(patient, d_sys)
        out = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=prev_hit,
        )
        hit = bool(out["hit"])
        prev_hit = hit

        d_pat = float(out["dist_ratio"]) * d_sys
        t_pat = float(out["t_pat"])
        if hit and t_pat > 0.01:
            observed_speeds.append(d_pat / t_pat)
        elif t_sys > 0.01:
            observed_speeds.append(d_pat / t_sys)

        # update model on executed trial
        model.update(hit, d_sys, t_sys)

        counts_5x5[i, j] += 1

        # rolling hit rate
        hit_list.append(int(hit))
        if len(hit_list) >= rolling_window:
            roll = float(np.mean(hit_list[-rolling_window:]))
        else:
            roll = float(np.mean(hit_list))

        # log
        hist["a_i"].append(i)
        hist["a_j"].append(j)
        hist["d"].append(d)
        hist["t"].append(t)
        hist["d_sys"].append(d_sys)
        hist["t_sys"].append(t_sys)
        hist["p_pred"].append(float(p_pred_intended))
        hist["uncert"].append(float(uncert_val))
        hist["hit"].append(int(hit))
        hist["rolling_hit"].append(roll)
        hist["beta0"].append(float(model.beta0))
        hist["beta_d"].append(float(model.beta_d))
        hist["beta_t"].append(float(model.beta_t))

    return hist, counts_5x5, patient
