from __future__ import annotations
import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class StaircaseConfig:
    # Bounds (match what's realistic in your game)
    d_min: float = 0.10
    d_max: float = 0.80
    t_min: float = 1.0
    t_max: float = 7.0

    # Step sizes
    d_step: float = 0.05         # meters per difficulty change
    t_step: float = 0.1          # seconds per difficulty change

    # Streak thresholds (classic "k-up / k-down" staircase)
    k_up: int = 2                # after 2 consecutive hits -> harder
    k_down: int = 2              # after 2 consecutive misses -> easier

    # Optional weights
    d_weight: float = 1.0
    t_weight: float = 1.0

    # Quantize to a grid so values don't jitter with floats
    quantize_d: float = 0.05
    quantize_t: float = 0.1


class StaircaseController:
    """
    Streak-based staircasing controller:
      - Consecutive successes => harder
      - Consecutive failures  => easier

    Keeps internal streak counts and updates (d, t) only when a threshold is reached.
    Each harder/easier update randomizes the axis and changes only one variable.
    """

    def __init__(self, config: StaircaseConfig, d0: float = 0.30, t0: float = 4.0):
        self.cfg = config
        self.d = clamp(d0, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(t0, self.cfg.t_min, self.cfg.t_max)
        self.success_streak = 0
        self.fail_streak = 0

    def _quantize(self, d: float, t: float) -> Tuple[float, float]:
        d = round(d / self.cfg.quantize_d) * self.cfg.quantize_d
        t = round(t / self.cfg.quantize_t) * self.cfg.quantize_t
        return d, t

    def _pick_axis(self) -> str:
        return "d" if random.randint(0, 1) == 0 else "t"

    def _make_harder(self) -> None:
        axis = self._pick_axis()
        if axis == "d":
            self.d += self.cfg.d_weight * self.cfg.d_step
        else:
            self.t -= self.cfg.t_weight * self.cfg.t_step

        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(self.t, self.cfg.t_min, self.cfg.t_max)
        self.d, self.t = self._quantize(self.d, self.t)

    def _make_easier(self) -> None:
        axis = self._pick_axis()
        if axis == "d":
            self.d -= self.cfg.d_weight * self.cfg.d_step
        else:
            self.t += self.cfg.t_weight * self.cfg.t_step

        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(self.t, self.cfg.t_min, self.cfg.t_max)
        self.d, self.t = self._quantize(self.d, self.t)

    def update(self, hit: bool) -> Tuple[float, float, str]:
        if hit:
            self.success_streak += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
            self.success_streak = 0

        if self.success_streak >= self.cfg.k_up:
            self._make_harder()
            self.success_streak = 0
            return self.d, self.t, "harder"

        if self.fail_streak >= self.cfg.k_down:
            self._make_easier()
            self.fail_streak = 0
            return self.d, self.t, "easier"

        return self.d, self.t, "none"

    # Sample a random pair (d,t) along the diagonal of the (d,t) matrix
    def sample_diagonal_pair(self, cfg: StaircaseConfig) -> Tuple[float, float]:
        """
        Sample a (d,t) pair along the diagonal:
        larger d <-> larger t
        """
        rng = np.random.default_rng()
        u = rng.uniform(0, 1)
        d = self.cfg.d_min + u * (self.cfg.d_max - self.cfg.d_min)
        t = self.cfg.t_min + u * (self.cfg.t_max - self.cfg.t_min)

        # quantize to grid
        d = round(d / self.cfg.d_step) * self.cfg.d_step
        t = round(t / self.cfg.t_step) * self.cfg.t_step

        return d, t


def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    """
    Your patient model includes patient.d_means which define distance bins.
    Map continuous d_sys to an integer level PatientModel expects.
    """
    d_means = np.asarray(patient.d_means, dtype=float)
    idx = np.where(d_means <= d_sys)[0]
    return int(idx[-1]) if len(idx) else 0


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
    Falls back to StaircaseConfig defaults if calibration data is insufficient.
    """
    ABS_D_MIN, ABS_D_MAX = 0.05, 1.5
    ABS_T_MIN, ABS_T_MAX = 0.15, 15.0

    defaults = StaircaseConfig()

    if not calibration_result:
        return defaults.d_min, defaults.d_max, defaults.t_min, defaults.t_max

    trials = calibration_result.get("trials", [])
    speeds = []
    for tr in trials:
        hit = tr.get("hit", tr.get("reached", False))
        t_pat = float(tr.get("t_pat_obs", tr.get("t_pat", 0)))
        d_sys = float(tr.get("d_sys", 0))
        if hit and t_pat > 0.01 and d_sys > 0.01:
            speeds.append(d_sys / t_pat)

    if len(speeds) < 3:
        return defaults.d_min, defaults.d_max, defaults.t_min, defaults.t_max

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


# ============================================================
# 5x5 binning (match operations_research.py: split range into fifths)
# ============================================================

BIN_NAMES_5 = ["closest/shortest", "close/short", "medium", "far/long", "farthest/longest"]


def level5(x: float, xmin: float, xmax: float) -> int:
    """
    Map x in [xmin,xmax] -> {0,1,2,3,4} by fifths.
    Bin 0 is smallest value (closest/shortest), Bin 4 is largest (farthest/longest).
    """
    if xmax <= xmin:
        return 0
    r = (x - xmin) / (xmax - xmin)
    r = clamp(r, 0.0, 1.0)
    # Fifths: [0,.2)->0, [.2,.4)->1, ...
    return int(min(4, math.floor(5 * r + 1e-12)))


def bin25(d: float, t: float, cfg: StaircaseConfig) -> Tuple[int, int]:
    """
    Returns (dist_bin, time_bin) in {0..4}x{0..4}.
    dist_bin 0=closest, 4=farthest; time_bin 0=shortest, 4=longest.
    """
    i = level5(d, cfg.d_min, cfg.d_max)
    j = level5(t, cfg.t_min, cfg.t_max)
    return i, j


def count_trials_per_bin(
    d_list: List[float],
    t_list: List[float],
    cfg: StaircaseConfig,
) -> np.ndarray:
    """
    Count trials for each (d,t) bin in a 5x5 grid.
    Rows=distance bins (closest->farthest), cols=time bins (shortest->longest).
    """
    counts = np.zeros((5, 5), dtype=int)
    for d, t in zip(d_list, t_list):
        i, j = bin25(float(d), float(t), cfg)
        counts[i, j] += 1
    return counts


# ============================================================
# Simulation
# ============================================================


def run_sim(
    patient: PatientModel,
    n_trials: int = 2000,
    seed: int = 7,
    d0: float = 0.30,
    t0: float = 4.0,
    cfg: StaircaseConfig | None = None,
    calibration: bool = True,
) -> Dict[str, List]:
    rng = np.random.default_rng(seed)

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)

    if cfg is None:
        cal_d_min, cal_d_max, cal_t_min, cal_t_max = derive_bounds_from_calibration(
            calibration_result, patient)
        cfg = StaircaseConfig(
            d_min=cal_d_min, d_max=cal_d_max,
            t_min=cal_t_min, t_max=cal_t_max,
        )

    controller = StaircaseController(cfg, d0=d0, t0=t0)

    logs: Dict[str, List] = {
        "trial": [],
        "d": [],
        "t": [],
        "direction": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "t_pat": [],
        "d_pat": [],
        "action": [],
    }

    previous_hit = True
    observed_speeds = []
    for k in range(n_trials):
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_d_min, new_d_max, new_t_min, new_t_max, changed = expand_bounds_if_needed(
                cfg.d_min, cfg.d_max, cfg.t_min, cfg.t_max, observed_speeds)
            if changed:
                cfg.d_min, cfg.d_max = new_d_min, new_d_max
                cfg.t_min, cfg.t_max = new_t_min, new_t_max

        d_sys, t_sys = controller.d, controller.t
        lvl = distance_level_from_patient_bins(patient, d_sys)
        direction = int(rng.integers(0, 5))

        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
            direction_bin=direction,
        )

        hit = bool(outcome["hit"])
        previous_hit = hit

        # Update controller based on hit/miss streaks
        _, _, action = controller.update(hit)

        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])

        if hit and t_pat > 0.01:
            observed_speeds.append(d_pat / t_pat)
        elif t_sys > 0.01:
            observed_speeds.append(d_pat / t_sys)

        logs["trial"].append(k)
        logs["d"].append(float(d_sys))
        logs["t"].append(float(t_sys))
        logs["direction"].append(direction)
        logs["hit"].append(int(hit))
        logs["time_ratio"].append(float(outcome["time_ratio"]))
        logs["dist_ratio"].append(float(outcome["dist_ratio"]))
        logs["t_pat"].append(float(t_pat))
        logs["d_pat"].append(float(d_pat))
        logs["action"].append(action)

    counts = count_trials_per_bin(
        d_list=logs["d"],
        t_list=logs["t"],
        cfg=cfg,
    )

    return logs, counts
