from __future__ import annotations
# rl_logreg_distance_only.py
# Distance-index + logistic regression with fixed time.

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
# Candidate grid (distance only; time fixed)
# -----------------------------
D_MIN, D_MAX = 0.10, 0.80
T_FIXED_DEFAULT = 5.0

D_STEP = 0.05
D_GRID = np.round(np.arange(D_MIN, D_MAX + 1e-9, D_STEP), 4)


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


def bin25(d, t, d_min, d_max, t_fixed):
    return (level5(d, d_min, d_max), level5(t, t_fixed, t_fixed))


def pick_with_tiebreak(candidates, score_fn, rng, maximize=False, atol=1e-12):
    scores = np.array([float(score_fn(c)) for c in candidates], dtype=float)
    best_score = float(np.max(scores) if maximize else np.min(scores))
    tie_idx = np.where(np.isclose(scores, best_score, atol=atol, rtol=0.0))[0]
    chosen = int(rng.choice(tie_idx))
    return candidates[chosen]


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


def derive_bounds_from_calibration(calibration_result, patient, t_fixed=T_FIXED_DEFAULT):
    ABS_D_MIN, ABS_D_MAX = 0.05, 1.5

    if not calibration_result:
        return D_MIN, D_MAX, float(t_fixed), float(t_fixed)

    trials = calibration_result.get("trials", [])
    speeds = []
    for tr in trials:
        hit = tr.get("hit", tr.get("reached", False))
        t_pat = float(tr.get("t_pat_obs", tr.get("t_pat", 0)))
        d_sys = float(tr.get("d_sys", 0))
        if hit and t_pat > 0.01 and d_sys > 0.01:
            speeds.append(d_sys / t_pat)

    if len(speeds) < 3:
        return D_MIN, D_MAX, float(t_fixed), float(t_fixed)

    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = ABS_D_MIN

    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15

    return float(d_min_cal), float(min(d_max_cal, ABS_D_MAX)), float(t_fixed), float(t_fixed)


def expand_bounds_if_needed(d_min, d_max, t_min, t_max, observed_speeds, t_fixed=T_FIXED_DEFAULT):
    # In 1-var mode time is fixed.
    return d_min, d_max, float(t_fixed), float(t_fixed), False


def apply_jitter(d, t, rng, p_jitter=0.7, d_sigma=0.06, d_min=D_MIN, d_max=D_MAX):
    d_sys, t_sys = d, t
    if rng.random() < p_jitter:
        eps = rng.normal(0.0, d_sigma)
        d_sys = float(np.clip(d * (1.0 + eps), d_min, d_max))
    return d_sys, t_sys


def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_d(d, d_min=D_MIN, d_max=D_MAX):
    return (d - d_min) / (d_max - d_min + 1e-12)


INDEX_W_D = 1.0


class DifficultyIndexLogReg:
    """
    p_hat = sigmoid(beta0 + beta1 * I)
    I = INDEX_W_D * d_norm
    """

    def __init__(
        self,
        rng,
        beta0=0.0,
        beta1=1.0,
        lr_beta=0.25,
        p_star=0.70,
        lam_p=0.55,
        d_min=D_MIN,
        d_max=D_MAX,
    ):
        self.rng = rng
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)

        self.w_d = float(INDEX_W_D)
        self.w_t = 0.0

        self.lr_beta = float(lr_beta)

        self.p_star = float(p_star)
        self.lam_p = float(lam_p)
        self.d_min = float(d_min)
        self.d_max = float(d_max)

        self._seen = set()

    def index(self, d, t):
        dn = normalize_d(d, self.d_min, self.d_max)
        ht = 0.0
        return self.w_d * dn, dn, ht

    def predict_p(self, d, t):
        if len(self._seen) < 2:
            return 0.5
        I, _, _ = self.index(d, t)
        return float(sigmoid(self.beta0 + self.beta1 * I))

    def uncertainty(self, d, t):
        p = self.predict_p(d, t)
        return 1.0 - 2.0 * abs(p - 0.5)

    def update(self, hit, d_exec, t_exec, d_intended, t_intended):
        y = 1.0 if hit else 0.0
        self._seen.add(int(y))

        I_exec, _, _ = self.index(d_exec, t_exec)
        p_exec = sigmoid(self.beta0 + self.beta1 * I_exec)

        g = p_exec - y
        self.beta0 -= self.lr_beta * g
        self.beta1 -= self.lr_beta * g * I_exec

        self.beta1 = float(np.clip(self.beta1, -8.0, 8.0))
        self.beta0 = float(np.clip(self.beta0, -12.0, 12.0))

        return float(p_exec), float(I_exec), None


def run_controller(
    patient: PatientModel,
    n_trials=200,
    seed=7,
    p_star=0.70,
    p_tol=0.05,
    explore_prob=0.10,
    p_jitter=0.30,
    t_fixed=T_FIXED_DEFAULT,
    calibration=True,
):
    rng = np.random.default_rng(seed)
    t_fixed = float(t_fixed)

    cur_d_min, cur_d_max = D_MIN, D_MAX

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        cur_d_min, cur_d_max, _cur_t_min, _cur_t_max = derive_bounds_from_calibration(
            calibration_result, patient, t_fixed=t_fixed
        )

    model = DifficultyIndexLogReg(
        rng=rng,
        beta0=0.0,
        beta1=1.0,
        lr_beta=0.30,
        p_star=p_star,
        lam_p=0.55,
        d_min=cur_d_min,
        d_max=cur_d_max,
    )

    counts = np.zeros((5, 5), dtype=int)

    hist = {
        "a_i": [],
        "a_j": [],
        "d": [],
        "t": [],
        "d_sys": [],
        "t_sys": [],
        "p_pred_intended": [],
        "p_pred_exec": [],
        "uncert": [],
        "hit": [],
        "rolling_hit": [],
        "beta0": [],
        "beta1": [],
        "w_d": [],
        "w_t": [],
        "index_exec": [],
    }

    prev_hit = True
    rolling_window = 20
    hit_list = []

    if calibration_result:
        for trial in calibration_result.get("trials", []):
            d_exec = float(trial.get("d_sys", cur_d_min))
            hit = bool(trial.get("hit", False))
            model.update(hit, d_exec, t_fixed, d_exec, t_fixed)

    local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
    local_candidates = [(float(d), t_fixed) for d in local_d_grid]

    observed_speeds = []

    for k in range(n_trials):
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_d_min, new_d_max, _new_t_min, _new_t_max, changed = expand_bounds_if_needed(
                cur_d_min, cur_d_max, t_fixed, t_fixed, observed_speeds, t_fixed=t_fixed
            )
            if changed:
                cur_d_min, cur_d_max = new_d_min, new_d_max
                model.d_min, model.d_max = cur_d_min, cur_d_max
                local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
                local_candidates = [(float(d), t_fixed) for d in local_d_grid]

        scored = []
        for d, t in local_candidates:
            p_pred = model.predict_p(d, t)
            uncert = 1.0 - 2.0 * abs(p_pred - 0.5)
            scored.append((d, t, p_pred, uncert))

        near_target = [x for x in scored if abs(x[2] - p_star) <= p_tol]
        pool = near_target if len(near_target) > 0 else scored

        if rng.random() < explore_prob:
            d, t, p_pred_intended, uncert_val = pick_with_tiebreak(
                pool, score_fn=lambda x: x[3], rng=rng, maximize=True
            )
        else:
            d, t, p_pred_intended, uncert_val = pick_with_tiebreak(
                pool, score_fn=lambda x: abs(x[2] - p_star), rng=rng, maximize=False
            )

        i, j = bin25(d, t, cur_d_min, cur_d_max, t_fixed)

        d_sys, t_sys = apply_jitter(
            d,
            t,
            rng,
            p_jitter=p_jitter,
            d_min=cur_d_min,
            d_max=cur_d_max,
        )

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

        p_exec_before = model.predict_p(d_sys, t_sys)
        p_int_before = p_pred_intended
        p_exec_after, I_exec, p_int_after = model.update(hit, d_sys, t_sys, d, t)

        counts[i, j] += 1

        hit_list.append(int(hit))
        if len(hit_list) >= rolling_window:
            roll = float(np.mean(hit_list[-rolling_window:]))
        else:
            roll = float(np.mean(hit_list))

        hist["a_i"].append(i)
        hist["a_j"].append(j)
        hist["d"].append(d)
        hist["t"].append(t)
        hist["d_sys"].append(d_sys)
        hist["t_sys"].append(t_sys)
        hist["p_pred_intended"].append(float(p_int_before))
        hist["p_pred_exec"].append(float(p_exec_before))
        hist["uncert"].append(float(uncert_val))
        hist["hit"].append(int(hit))
        hist["rolling_hit"].append(roll)
        hist["beta0"].append(float(model.beta0))
        hist["beta1"].append(float(model.beta1))
        hist["w_d"].append(float(model.w_d))
        hist["w_t"].append(float(model.w_t))
        hist["index_exec"].append(float(I_exec))

    return hist, counts, patient
