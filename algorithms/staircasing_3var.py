from __future__ import annotations
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
    # Bounds (match what’s realistic in your game)
    d_min: float = 0.10
    d_max: float = 0.80
    t_min: float = 1.0
    t_max: float = 7.0
    dir_min: int = 0
    dir_max: int = 8

    # Step sizes
    d_step: float = 0.05         # meters per difficulty change
    t_step: float = 0.1          # seconds per difficulty change
    dir_step: int = 1            # direction bin per change

    # Streak thresholds (classic “k-up / k-down” staircase)
    k_up: int = 3                 # after 2 consecutive hits -> harder
    k_down: int = 2               # after 1 consecutive miss -> easier

    # How to change difficulty when moving "harder" or "easier"
    couple_distance_and_time: bool = True

    # Optional weights
    d_weight: float = 1.0
    t_weight: float = 1.0

    # Quantize to a grid so values don’t jitter with floats
    quantize_d: float = 0.05
    quantize_t: float = 0.1
    quantize_dir: int = 1

class StaircaseController:
    """
    Streak-based staircasing controller:
      - Consecutive successes => harder
      - Consecutive failures  => easier

    Keeps internal streak counts and updates (d, t) only when a threshold is reached.
    """

    def __init__(
        self,
        config: StaircaseConfig,
        d0: float = 0.30,
        t0: float = 4.0,
        dir0: int = 0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = config
        self.d = clamp(d0, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(t0, self.cfg.t_min, self.cfg.t_max)
        self.dir = int(clamp(dir0, self.cfg.dir_min, self.cfg.dir_max))
        self.success_streak = 0
        self.fail_streak = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    def _quantize(self, d: float, t: float, direction: int) -> Tuple[float, float, int]:
        d = round(d / self.cfg.quantize_d) * self.cfg.quantize_d
        t = round(t / self.cfg.quantize_t) * self.cfg.quantize_t
        direction = int(round(direction / self.cfg.quantize_dir) * self.cfg.quantize_dir)
        return d, t, direction

    def _rank_dirs(self, patient: PatientModel) -> List[int]:
        alpha = np.asarray(patient.spatial_success_alpha, dtype=float)
        beta = np.asarray(patient.spatial_success_beta, dtype=float)
        p_hit = alpha / (alpha + beta + 1e-12)
        # Lower p_hit => harder. Stable sort so ties keep deterministic order.
        return list(np.argsort(p_hit, kind="stable"))

    def _move_dir(self, patient: PatientModel, harder: bool) -> None:
        order = self._rank_dirs(patient)
        try:
            idx = order.index(self.dir)
        except ValueError:
            idx = 0
        if harder:
            idx = max(0, idx - 1)
        else:
            idx = min(len(order) - 1, idx + 1)
        self.dir = int(order[idx])

    def _sample_adjust_var(self) -> str:
        return "d" if float(self.rng.uniform()) < 0.5 else "t"

    def sample_direction_ts(self, patient: PatientModel) -> int:
        alpha = np.asarray(patient.spatial_success_alpha, dtype=float)
        beta = np.asarray(patient.spatial_success_beta, dtype=float)
        ts_samples = self.rng.beta(alpha, beta)
        direction = int(np.argmax(ts_samples))
        direction = int(clamp(direction, self.cfg.dir_min, self.cfg.dir_max))
        self.dir = direction
        return direction

    def _make_harder(self, patient: PatientModel) -> None:
        adjust_var = self._sample_adjust_var()
        if adjust_var == "d":
            self.d += self.cfg.d_weight * self.cfg.d_step
        else:
            self.t -= self.cfg.t_weight * self.cfg.t_step

        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(self.t, self.cfg.t_min, self.cfg.t_max)
        self.dir = int(clamp(self.dir, self.cfg.dir_min, self.cfg.dir_max))
        self.d, self.t, self.dir = self._quantize(self.d, self.t, self.dir)

    def _make_easier(self, patient: PatientModel) -> None:
        adjust_var = self._sample_adjust_var()
        if adjust_var == "d":
            self.d -= self.cfg.d_weight * self.cfg.d_step
        else:
            self.t += self.cfg.t_weight * self.cfg.t_step

        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(self.t, self.cfg.t_min, self.cfg.t_max)
        self.dir = int(clamp(self.dir, self.cfg.dir_min, self.cfg.dir_max))
        self.d, self.t, self.dir = self._quantize(self.d, self.t, self.dir)

    def update(self, hit: bool, patient: PatientModel) -> Tuple[float, float, int, str]:
        if hit:
            self.success_streak += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
            self.success_streak = 0

        if self.success_streak >= self.cfg.k_up:
            self._make_harder(patient)
            self.success_streak = 0
            return self.d, self.t, self.dir, "harder"

        if self.fail_streak >= self.cfg.k_down:
            self._make_easier(patient)
            self.fail_streak = 0
            return self.d, self.t, self.dir, "easier"

        return self.d, self.t, self.dir, "none"
    
    # Sample a random pair (d,t) along the diagonal of the (d,t) matrix
    def sample_diagonal_pair(self, cfg: StaircaseConfig) -> Tuple[float, float, int]:
        """
        Sample a (d,t) pair along the diagonal:
        larger d <-> larger t
        """
        rng = np.random.default_rng()
        u = rng.uniform(0, 1)
        d = self.cfg.d_min + u * (self.cfg.d_max - self.cfg.d_min)
        t = self.cfg.t_min + u * (self.cfg.t_max - self.cfg.t_min)
        direction = int(rng.integers(cfg.dir_min, cfg.dir_max + 1))

        # quantize to grid
        d = round(d / self.cfg.d_step) * self.cfg.d_step
        t = round(t / self.cfg.t_step) * self.cfg.t_step

        return d, t, direction

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
        idx = int(np.clip(direction, 0, 8))
        n_reached = float(stats.get("n_reached", 0))
        n_censored = float(stats.get("n_censored", 0))
        patient.spatial_success_alpha[idx] += n_reached
        patient.spatial_success_beta[idx] += n_censored


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
    dir0: int = 0,
    cfg: StaircaseConfig | None = None,
    calibration: bool = True,
) -> Dict[str, List]:
    cfg = cfg or StaircaseConfig()
    rng = np.random.default_rng(seed)
    controller = StaircaseController(cfg, d0=d0, t0=t0, dir0=dir0, rng=rng)

    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)

    logs: Dict[str, List] = {
        "trial": [],
        "d": [],
        "t": [],
        "dir": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "t_pat": [],
        "d_pat": [],
        "action": [],
        "direction": [],
    }

    previous_hit = True
    for k in range(n_trials):
        resample = False
        d_sys, t_sys = controller.d, controller.t
        direction = controller.sample_direction_ts(patient)
        lvl = distance_level_from_patient_bins(patient, d_sys)

        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
            direction_bin=direction,
        )

        hit = bool(outcome["hit"])
        previous_hit = hit
        if hit:
            patient.spatial_success_alpha[direction] += 1.0
        else:
            patient.spatial_success_beta[direction] += 1.0

        # Update controller based on hit/miss streaks
        _, _, _, action = controller.update(hit, patient)
        if resample:
            action = f"resample: d,t,dir={controller.d:.2f},{controller.t:.2f},{controller.dir}"

        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])

        logs["trial"].append(k)
        logs["d"].append(float(d_sys))
        logs["t"].append(float(t_sys))
        logs["dir"].append(int(direction))
        logs["hit"].append(int(hit))
        logs["time_ratio"].append(float(outcome["time_ratio"]))
        logs["dist_ratio"].append(float(outcome["dist_ratio"]))
        logs["t_pat"].append(float(t_pat))
        logs["d_pat"].append(float(d_pat))
        logs["action"].append(action)
        logs["direction"].append(int(direction))

    counts = count_trials_per_bin(
        d_list=logs["d"],
        t_list=logs["t"],
        cfg=cfg,
    )

    return logs, counts, patient
