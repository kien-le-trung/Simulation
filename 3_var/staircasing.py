from __future__ import annotations
import importlib.util
import math
import random
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
PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v3.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v3", PATIENT_SIM_PATH)
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

    # Step sizes
    d_step: float = 0.05         # meters per difficulty change
    t_step: float = 0.1          # seconds per difficulty change

    # Streak thresholds (classic “k-up / k-down” staircase)
    k_up: int = 2                 # after 2 consecutive hits -> harder
    k_down: int = 2               # after 1 consecutive miss -> easier

    # How to change difficulty when moving "harder" or "easier"
    couple_distance_and_time: bool = True

    # Optional weights
    d_weight: float = 1.0
    t_weight: float = 1.0

    # Quantize to a grid so values don’t jitter with floats
    quantize_d: float = 0.05
    quantize_t: float = 0.1


class StaircaseController:
    """
    Streak-based staircasing controller:
      - Consecutive successes => harder
      - Consecutive failures  => easier

    Keeps internal streak counts and updates (d, t) only when a threshold is reached.
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

    def _make_harder(self) -> None:
        if self.cfg.couple_distance_and_time:
            self.d += self.cfg.d_weight * self.cfg.d_step
            self.t -= self.cfg.t_weight * self.cfg.t_step
        else:
            if random.randint(0, 1) == 0:
                self.t -= self.cfg.t_weight * self.cfg.t_step
            else:
                self.d += self.cfg.d_weight * self.cfg.d_step

        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(self.t, self.cfg.t_min, self.cfg.t_max)
        self.d, self.t = self._quantize(self.d, self.t)

    def _make_easier(self) -> None:
        if self.cfg.couple_distance_and_time:
            self.d -= self.cfg.d_weight * self.cfg.d_step
            self.t += self.cfg.t_weight * self.cfg.t_step
        else:
            if random.randint(0, 1) == 0:
                self.t += self.cfg.t_weight * self.cfg.t_step
            else:
                self.d -= self.cfg.d_weight * self.cfg.d_step

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
    n_trials: int = 150,
    seed: int = 7,
    d0: float = 0.30,
    t0: float = 4.0,
    cfg: StaircaseConfig | None = None,
) -> Dict[str, List]:
    cfg = cfg or StaircaseConfig()
    controller = StaircaseController(cfg, d0=d0, t0=t0)

    logs: Dict[str, List] = {
        "trial": [],
        "d_sys": [],
        "t_sys": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "t_pat": [],
        "d_pat": [],
        "action": [],
    }

    previous_hit = True
    for k in range(n_trials):
        resample = False
        if k % 10 == 0 and k > 0:
            resample = True
            controller.d, controller.t = controller.sample_diagonal_pair(cfg)
            controller.success_streak = 0
            controller.fail_streak = 0
        d_sys, t_sys = controller.d, controller.t
        lvl = distance_level_from_patient_bins(patient, d_sys)

        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
        )

        hit = bool(outcome["hit"])
        previous_hit = hit

        # Update controller based on hit/miss streaks
        _, _, action = controller.update(hit)
        if resample:
            action = f"resample: d,t={controller.d:.2f},{controller.t:.2f}"

        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])

        logs["trial"].append(k)
        logs["d_sys"].append(float(d_sys))
        logs["t_sys"].append(float(t_sys))
        logs["hit"].append(int(hit))
        logs["time_ratio"].append(float(outcome["time_ratio"]))
        logs["dist_ratio"].append(float(outcome["dist_ratio"]))
        logs["t_pat"].append(float(t_pat))
        logs["d_pat"].append(float(d_pat))
        logs["action"].append(action)

    counts = count_trials_per_bin(
        d_list=logs["d_sys"],
        t_list=logs["t_sys"],
        cfg=cfg,
    )

    return logs, counts