from __future__ import annotations
"""
Continuous-margin control for selecting distance with fixed time.

Key idea:
- Convert each trial into a continuous distance margin m_k = dist_ratio - 1.
- Track EWMA of margin: m_hat.
- PI controller pushes a scalar difficulty u in [0,1].
- Map u -> d_sys and keep t_sys fixed.
- Keep exploration/diversity and calibration logic from 2-var version.
"""

import math
import importlib.util
from pathlib import Path

import numpy as np

# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel
# ============================================================

T_FIXED_DEFAULT = 5.0

hit_margin = []
miss_margin = []


# ----------------------------
# Helpers: bins + rolling stats
# ----------------------------
def rolling_mean(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def level5(x, xmin, xmax):
    """Map x in [xmin,xmax] -> {0..4} by fifths."""
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


def bin25(d, t, dmin, dmax, tmin, tmax):
    return (level5(d, dmin, dmax), level5(t, tmin, tmax))


def rarity_bonus(counts_5x5, i, j, eps=1e-12):
    """-log(freq) with Laplace smoothing, higher for under-sampled bins."""
    total = counts_5x5.sum()
    freq = (counts_5x5[i, j] + 1.0) / (total + 25.0)
    return -math.log(freq + eps)


def distance_level_3(d, dmin, dmax):
    """Map d to distance_level in {0,1,2} (PatientModel clips to 0..2)."""
    u = (d - dmin) / (dmax - dmin + 1e-12)
    if u < 1 / 3:
        return 0
    if u < 2 / 3:
        return 1
    return 2


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
    """
    Derive generous (d_min, d_max, t_fixed, t_fixed, v_min, v_max) from calibration.
    """
    ABS_D_MIN, ABS_D_MAX = 0.05, 1.5

    FALLBACK = (0.10, 1.0, float(t_fixed), float(t_fixed), 0.10, 1.50)

    if not calibration_result:
        return FALLBACK

    trials = calibration_result.get("trials", [])
    speeds = []
    for tr in trials:
        hit = tr.get("hit", tr.get("reached", False))
        t_pat = float(tr.get("t_pat_obs", tr.get("t_pat", 0)))
        d_sys = float(tr.get("d_sys", 0))
        if hit and t_pat > 0.01 and d_sys > 0.01:
            speeds.append(d_sys / t_pat)

    if len(speeds) < 3:
        return FALLBACK

    speeds = np.array(speeds)
    v_slow = max(float(np.percentile(speeds, 10)), 0.01)
    v_fast = max(float(np.percentile(speeds, 90)), v_slow * 1.5)

    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = ABS_D_MIN

    v_min_cal = max(0.01, v_slow * 0.5)
    v_max_cal = v_fast * 2.0

    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15

    return (
        float(d_min_cal),
        float(min(d_max_cal, ABS_D_MAX)),
        float(t_fixed),
        float(t_fixed),
        float(v_min_cal),
        float(v_max_cal),
    )


def expand_bounds_if_needed(d_min, d_max, t_min, t_max, v_min, v_max, observed_speeds, t_fixed=T_FIXED_DEFAULT):
    """
    Backup plan: keep time fixed and only expand speed bounds if needed.
    Returns (d_min, d_max, t_fixed, t_fixed, v_min, v_max, changed).
    """
    if len(observed_speeds) < 5:
        return d_min, d_max, float(t_fixed), float(t_fixed), v_min, v_max, False

    arr = np.array(observed_speeds[-50:])
    v_p5 = max(float(np.percentile(arr, 5)), 0.01)
    v_p95 = float(np.percentile(arr, 95))

    changed = False

    if v_p5 < v_min * 0.7:
        v_min = v_p5 * 0.5
        changed = True
    if v_p95 > v_max * 0.8:
        v_max = v_p95 * 2.0
        changed = True

    return d_min, d_max, float(t_fixed), float(t_fixed), v_min, v_max, changed


# ----------------------------
# Continuous-margin PI controller
# ----------------------------
class MarginPIController:
    """
    Controls difficulty u in [0,1] using distance-only continuous margin feedback.

    Margin definition:
      m = dist_ratio - 1
    """

    def __init__(self, *, m_star=0.0, Kp=0.9, Ki=0.15, ewma_alpha=0.15, u0=0.35):
        self.m_star = float(m_star)
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.alpha = float(ewma_alpha)

        self.u = float(np.clip(u0, 0.0, 1.0))
        self.m_hat = 0.0
        self.e_int = 0.0

    def update(self, m_k):
        self.m_hat = (1.0 - self.alpha) * self.m_hat + self.alpha * float(m_k)

        e = self.m_hat - self.m_star

        pushing_up = e > 0 and self.u >= 1.0 - 1e-9
        pushing_down = e < 0 and self.u <= 0.0 + 1e-9
        if not (pushing_up or pushing_down):
            self.e_int += e

        du = self.Kp * e + self.Ki * self.e_int
        self.u = float(np.clip(self.u + du, 0.0, 1.0))
        return self.u, self.m_hat, e


# ----------------------------
# Map difficulty u -> d (time fixed)
# ----------------------------
def propose_d_from_u(u, *, dmin=0.10, dmax=1.0):
    d_base = dmin + u * (dmax - dmin)
    return float(d_base)


def choose_with_exploration_and_diversity(
    d_base,
    *,
    t_fixed,
    counts_5x5,
    dmin,
    dmax,
    tmin,
    tmax,
    rng,
    n_candidates=25,
    d_sigma=0.05,
    w_close=1.0,
    w_rare=0.35,
):
    """
    Build a local candidate set around d_base with fixed t, then pick by:
    score = -w_close * normalized_distance_to_base + w_rare * rarity_bonus(bin)
    """
    cand = []
    scores = []

    t = float(t_fixed)
    for _ in range(n_candidates):
        d = float(np.clip(rng.normal(d_base, d_sigma), dmin, dmax))

        dn = (d - d_base) / (dmax - dmin + 1e-12)
        close_pen = dn * dn

        i, j = bin25(d, t, dmin, dmax, tmin, tmax)
        rare = rarity_bonus(counts_5x5, i, j)

        score = -w_close * close_pen + w_rare * rare
        cand.append((d, t, i, j))
        scores.append(score)

    s = np.array(scores, dtype=float)
    s = s - s.max()
    p = np.exp(s)
    p = p / p.sum()

    idx = int(rng.choice(len(cand), p=p))
    d, t, i, j = cand[idx]
    return d, t, i, j


def sample_diagonal_pair(*, dmin, dmax, t_fixed, d_step, rng=None):
    """
    In 1-var mode this samples distance only and keeps time fixed.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.uniform(0.0, 1.0)
    d = dmin + u * (dmax - dmin)

    d = round(d / d_step) * d_step

    return float(d), float(t_fixed)


# ----------------------------
# Main simulation loop
# ----------------------------
def run_sim(
    patient: PatientModel,
    n_trials=100,
    seed=7,
    # controller targets/tuning
    m_star=0.0,
    Kp=0.9,
    Ki=0.15,
    ewma_alpha=0.15,
    # action space bounds (defaults; overridden by calibration)
    dmin=0.10,
    dmax=1.0,
    tmin=T_FIXED_DEFAULT,
    tmax=T_FIXED_DEFAULT,
    d_step=0.05,
    # mapping bounds for required speed (kept for compatibility/logging)
    vmin=0.10,
    vmax=1.50,
    # diagonal resample control
    diag_after=10,
    calibration=True,
    t_fixed=T_FIXED_DEFAULT,
):
    rng = np.random.default_rng(seed)

    t_fixed = float(t_fixed)
    tmin = t_fixed
    tmax = t_fixed

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        cal_dmin, cal_dmax, _cal_tmin, _cal_tmax, cal_vmin, cal_vmax = derive_bounds_from_calibration(
            calibration_result, patient, t_fixed=t_fixed
        )
        dmin, dmax = cal_dmin, cal_dmax
        vmin, vmax = cal_vmin, cal_vmax

    ctrl = MarginPIController(m_star=m_star, Kp=Kp, Ki=Ki, ewma_alpha=ewma_alpha, u0=0.35)

    counts_5x5 = np.zeros((5, 5), dtype=int)
    previous_hit = True

    hist = {
        "u": [],
        "d": [],
        "t": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "margin": [],
        "m_hat": [],
        "err": [],
        "bin_i": [],
        "bin_j": [],
    }

    observed_speeds = []

    for k in range(n_trials):
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_dmin, new_dmax, _new_tmin, _new_tmax, new_vmin, new_vmax, changed = expand_bounds_if_needed(
                dmin, dmax, tmin, tmax, vmin, vmax, observed_speeds, t_fixed=t_fixed
            )
            if changed:
                dmin, dmax = new_dmin, new_dmax
                vmin, vmax = new_vmin, new_vmax

        if diag_after is not None and (k % diag_after == 0):
            d_sys, t_sys = sample_diagonal_pair(
                dmin=dmin,
                dmax=dmax,
                t_fixed=t_fixed,
                d_step=d_step,
                rng=rng,
            )
            i, j = bin25(d_sys, t_sys, dmin, dmax, tmin, tmax)
        else:
            d_base = propose_d_from_u(ctrl.u, dmin=dmin, dmax=dmax)

            d_sys, t_sys, i, j = choose_with_exploration_and_diversity(
                d_base,
                t_fixed=t_fixed,
                counts_5x5=counts_5x5,
                dmin=dmin,
                dmax=dmax,
                tmin=tmin,
                tmax=tmax,
                rng=rng,
            )

        counts_5x5[i, j] += 1

        lvl = distance_level_3(d_sys, dmin, dmax)

        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
        )

        hit = bool(outcome["hit"])
        previous_hit = hit

        time_ratio = float(outcome["time_ratio"])
        dist_ratio = float(outcome["dist_ratio"])

        m_k = dist_ratio - 1.0
        if hit:
            hit_margin.append(m_k)
        else:
            miss_margin.append(m_k)

        d_pat = dist_ratio * d_sys
        t_pat = float(outcome["t_pat"])
        if hit and t_pat > 0.01:
            observed_speeds.append(d_pat / t_pat)
        elif t_sys > 0.01:
            observed_speeds.append(d_pat / t_sys)

        _u_next, m_hat, err = ctrl.update(m_k)

        if diag_after is not None and (k % diag_after == 0):
            ctrl.u = (d_sys - dmin) / (dmax - dmin + 1e-12)

        hist["u"].append(ctrl.u)
        hist["d"].append(d_sys)
        hist["t"].append(t_sys)
        hist["hit"].append(int(hit))
        hist["time_ratio"].append(time_ratio)
        hist["dist_ratio"].append(dist_ratio)
        hist["margin"].append(m_k)
        hist["m_hat"].append(m_hat)
        hist["err"].append(err)
        hist["bin_i"].append(i)
        hist["bin_j"].append(j)

    return hist, counts_5x5, patient
