from __future__ import annotations
# rl_logreg_distance_only.py
# Distance-only logistic regression with fixed time.

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
PATIENT_FIXED_T = {
    "overall_weak": 4.0,
    "overall_medium": 4.0,
    "overall_strong": 4.0,
    "highspeed_lowrom": 4.0,
    "lowspeed_highrom": 4.0,
}

D_STEP = 0.05
D_GRID = np.round(np.arange(D_MIN, D_MAX + 1e-9, D_STEP), 4)


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


def cap_distance_bounds(patient: PatientModel, d_min: float, d_max: float):
    patient_reach = float(getattr(patient, "max_reach", d_max))
    if not np.isfinite(patient_reach) or patient_reach <= 0:
        patient_reach = float(d_max)

    capped_min = max(float(d_min), 0.0)
    capped_max = float(min(d_max, patient_reach))
    if capped_max < capped_min:
        capped_max = capped_min
    return capped_min, capped_max


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


def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


class DifficultyIndexLogReg:
    """
    p_hat = sigmoid(beta0 + beta1 * d)
    """

    def __init__(
        self,
        rng,
        beta0=3.0,
        beta1=-5.0,
        lr_beta=0.25,
        p_star=0.70,
        lam_p=0.55,
    ):
        self.rng = rng
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)

        self.lr_beta = float(lr_beta)

        self.p_star = float(p_star)
        self.lam_p = float(lam_p)

        self._seen = set()

    def predict_p(self, d):
        if len(self._seen) < 2:
            return 0.5
        return float(sigmoid(self.beta0 + self.beta1 * float(d)))

    def uncertainty(self, d):
        p = self.predict_p(d)
        return 1.0 - 2.0 * abs(p - 0.5)

    def update(self, hit, d_exec):
        y = 1.0 if hit else 0.0
        self._seen.add(int(y))

        d_exec = float(d_exec)
        p_exec = sigmoid(self.beta0 + self.beta1 * d_exec)

        g = p_exec - y
        self.beta0 -= self.lr_beta * g
        self.beta1 -= self.lr_beta * g * d_exec

        self.beta1 = float(np.clip(self.beta1, -8.0, 8.0))
        self.beta0 = float(np.clip(self.beta0, -12.0, 12.0))

        return float(p_exec), d_exec


def run_controller(
    patient: PatientModel,
    n_trials=200,
    seed=7,
    p_star=0.70,
    p_tol=0.05,
    t_fixed=T_FIXED_DEFAULT,
    calibration=True,
    patient_profile: str | None = None,
):
    rng = np.random.default_rng(seed)
    if patient_profile is not None and patient_profile in PATIENT_FIXED_T:
        t_fixed = float(PATIENT_FIXED_T[patient_profile])
    else:
        t_fixed = float(t_fixed)

    cur_d_min, cur_d_max = D_MIN, D_MAX

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        cur_d_min, cur_d_max, _cur_t_min, _cur_t_max = derive_bounds_from_calibration(
            calibration_result, patient, t_fixed=t_fixed
        )
    cur_d_min, cur_d_max = cap_distance_bounds(patient, cur_d_min, cur_d_max)

    model = DifficultyIndexLogReg(
        rng=rng,
        beta0=3.0,
        beta1=-5.0,
        lr_beta=0.30,
        p_star=p_star,
        lam_p=0.55,
    )
    # Kept for API compatibility with other simulators; not used for selection any longer.
    counts = np.zeros((5, 5), dtype=int)

    hist = {
        "d": [],
        "t": [],
        "d_sys": [],
        "t_sys": [],
        "p_pred_intended": [],
        "p_pred_exec": [],
        "hit": [],
        "rolling_hit": [],
        "beta0": [],
        "beta1": [],
        "d_exec": [],
    }

    prev_hit = True
    rolling_window = 20
    hit_list = []

    if calibration_result:
        for trial in calibration_result.get("trials", []):
            d_exec = float(trial.get("d_sys", cur_d_min))
            hit = bool(trial.get("hit", False))
            model.update(hit, d_exec)

    local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
    local_candidates = [float(d) for d in local_d_grid]

    observed_speeds = []

    for k in range(n_trials):
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_d_min, new_d_max, _new_t_min, _new_t_max, changed = expand_bounds_if_needed(
                cur_d_min, cur_d_max, t_fixed, t_fixed, observed_speeds, t_fixed=t_fixed
            )
            if changed:
                cur_d_min, cur_d_max = new_d_min, new_d_max
                local_d_grid = np.round(np.arange(cur_d_min, cur_d_max + 1e-9, D_STEP), 4)
                local_candidates = [float(d) for d in local_d_grid]

        scored = []
        for d in local_candidates:
            p_pred = model.predict_p(d)
            scored.append((d, p_pred))

        near_target = [x for x in scored if abs(x[1] - p_star) <= p_tol]
        pool = near_target if len(near_target) > 0 else scored

        d, p_pred_intended = pick_with_tiebreak(
            pool,
            score_fn=lambda x: abs(x[1] - p_star),
            rng=rng,
            maximize=False,
        )
        
        d_sys = float(d)
        t_sys = float(t_fixed)

        lvl = distance_level_from_patient_bins(patient, d_sys)
        out = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=prev_hit,
            direction_bin=int(rng.integers(0, 5)),
        )
        hit = bool(out["hit"])
        prev_hit = hit

        d_pat = float(out["dist_ratio"]) * d_sys
        t_pat = float(out["t_pat"])
        if hit and t_pat > 0.01:
            observed_speeds.append(d_pat / t_pat)
        elif t_sys > 0.01:
            observed_speeds.append(d_pat / t_sys)

        p_exec_before = model.predict_p(d_sys)
        p_int_before = p_pred_intended
        p_exec_after, d_feature = model.update(hit, d_sys)

        hit_list.append(int(hit))
        if len(hit_list) >= rolling_window:
            roll = float(np.mean(hit_list[-rolling_window:]))
        else:
            roll = float(np.mean(hit_list))

        hist["d"].append(d)
        hist["t"].append(t_fixed)
        hist["d_sys"].append(d_sys)
        hist["t_sys"].append(t_sys)
        hist["p_pred_intended"].append(float(p_int_before))
        hist["p_pred_exec"].append(float(p_exec_before))
        hist["hit"].append(int(hit))
        hist["rolling_hit"].append(roll)
        hist["beta0"].append(float(model.beta0))
        hist["beta1"].append(float(model.beta1))
        hist["d_exec"].append(float(d_feature))

    return hist, counts, patient
