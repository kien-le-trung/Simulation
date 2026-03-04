from __future__ import annotations
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


T_FIXED = 5.0


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class StaircaseConfig:
    d_min: float = 0.10
    d_max: float = 0.80
    t_min: float = T_FIXED
    t_max: float = T_FIXED

    d_step: float = 0.05
    t_step: float = 0.1

    k_up: int = 2
    k_down: int = 2

    d_weight: float = 1.0
    t_weight: float = 1.0

    quantize_d: float = 0.05
    quantize_t: float = 0.1


class StaircaseController:
    """
    Streak-based staircasing controller:
      - Consecutive successes => harder
      - Consecutive failures  => easier

    In 1-var mode, only distance is adjusted and time is fixed.
    """

    def __init__(self, config: StaircaseConfig, d0: float = 0.30, t0: float = T_FIXED):
        self.cfg = config
        self.d = clamp(d0, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(float(t0), self.cfg.t_min, self.cfg.t_max)
        self.success_streak = 0
        self.fail_streak = 0

    def _quantize(self, d: float, t: float) -> Tuple[float, float]:
        d = round(d / self.cfg.quantize_d) * self.cfg.quantize_d
        t = clamp(float(t), self.cfg.t_min, self.cfg.t_max)
        return d, t

    def _make_harder(self) -> None:
        self.d += self.cfg.d_weight * self.cfg.d_step
        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.d, self.t = self._quantize(self.d, self.t)

    def _make_easier(self) -> None:
        self.d -= self.cfg.d_weight * self.cfg.d_step
        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
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

    def sample_diagonal_pair(self, cfg: StaircaseConfig) -> Tuple[float, float]:
        """
        In 1-var mode, sample only distance and keep time fixed.
        """
        rng = np.random.default_rng()
        u = rng.uniform(0, 1)
        d = self.cfg.d_min + u * (self.cfg.d_max - self.cfg.d_min)
        d = round(d / self.cfg.d_step) * self.cfg.d_step
        return float(d), float(self.cfg.t_min)


def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
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


def derive_bounds_from_calibration(calibration_result, patient, t_fixed: float = T_FIXED):
    """
    Derive distance bounds from calibration data and keep time fixed at 5 s.
    """
    ABS_D_MIN, ABS_D_MAX = 0.05, 1.5

    defaults = StaircaseConfig()

    if not calibration_result:
        return defaults.d_min, defaults.d_max, float(t_fixed), float(t_fixed)

    trials = calibration_result.get("trials", [])
    speeds = []
    for tr in trials:
        hit = tr.get("hit", tr.get("reached", False))
        t_pat = float(tr.get("t_pat_obs", tr.get("t_pat", 0)))
        d_sys = float(tr.get("d_sys", 0))
        if hit and t_pat > 0.01 and d_sys > 0.01:
            speeds.append(d_sys / t_pat)

    if len(speeds) < 3:
        return defaults.d_min, defaults.d_max, float(t_fixed), float(t_fixed)

    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = ABS_D_MIN

    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15

    return float(d_min_cal), float(min(d_max_cal, ABS_D_MAX)), float(t_fixed), float(t_fixed)


def expand_bounds_if_needed(d_min, d_max, t_min, t_max, observed_speeds, t_fixed: float = T_FIXED):
    """
    Time is fixed in 1-var mode. Keep bounds unchanged.
    """
    return d_min, d_max, float(t_fixed), float(t_fixed), False


BIN_NAMES_5 = ["closest/shortest", "close/short", "medium", "far/long", "farthest/longest"]


def level5(x: float, xmin: float, xmax: float) -> int:
    if xmax <= xmin:
        return 0
    r = (x - xmin) / (xmax - xmin)
    r = clamp(r, 0.0, 1.0)
    return int(min(4, math.floor(5 * r + 1e-12)))


def bin25(d: float, t: float, cfg: StaircaseConfig) -> Tuple[int, int]:
    i = level5(d, cfg.d_min, cfg.d_max)
    j = level5(t, cfg.t_min, cfg.t_max)
    return i, j


def count_trials_per_bin(
    d_list: List[float],
    t_list: List[float],
    cfg: StaircaseConfig,
) -> np.ndarray:
    counts = np.zeros((5, 5), dtype=int)
    for d, t in zip(d_list, t_list):
        i, j = bin25(float(d), float(t), cfg)
        counts[i, j] += 1
    return counts


def run_sim(
    patient: PatientModel,
    n_trials: int = 2000,
    seed: int = 7,
    d0: float = 0.30,
    t0: float = T_FIXED,
    t_fixed: float | None = None,
    cfg: StaircaseConfig | None = None,
    calibration: bool = True,
) -> Dict[str, List]:
    if t_fixed is None:
        t_fixed = float(t0)
    else:
        t_fixed = float(t_fixed)
    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)

    if cfg is None:
        cal_d_min, cal_d_max, _cal_t_min, _cal_t_max = derive_bounds_from_calibration(
            calibration_result, patient, t_fixed=t_fixed
        )
        cfg = StaircaseConfig(
            d_min=cal_d_min,
            d_max=cal_d_max,
            t_min=t_fixed,
            t_max=t_fixed,
        )
    else:
        cfg.t_min = t_fixed
        cfg.t_max = t_fixed

    controller = StaircaseController(cfg, d0=d0, t0=t_fixed)

    logs: Dict[str, List] = {
        "trial": [],
        "d": [],
        "t": [],
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
                cfg.d_min, cfg.d_max, cfg.t_min, cfg.t_max, observed_speeds, t_fixed=t_fixed
            )
            if changed:
                cfg.d_min, cfg.d_max = new_d_min, new_d_max
                cfg.t_min, cfg.t_max = new_t_min, new_t_max

        d_sys, t_sys = controller.d, t_fixed
        controller.t = t_fixed
        lvl = distance_level_from_patient_bins(patient, d_sys)

        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
        )

        hit = bool(outcome["hit"])
        previous_hit = hit

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
