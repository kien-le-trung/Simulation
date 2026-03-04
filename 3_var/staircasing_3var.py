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
    d_max: float = 1.0
    t_min: float = 0.3
    t_max: float = 7.0
    dir_min: int = 0
    dir_max: int = 4

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
        dir0: int = 2,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = config
        self.d = clamp(d0, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(t0, self.cfg.t_min, self.cfg.t_max)
        self.dir = int(clamp(dir0, self.cfg.dir_min, self.cfg.dir_max))
        self.success_streak = 0
        self.fail_streak = 0
        self.rng = rng if rng is not None else np.random.default_rng()
        # Per-direction hit/miss counters (staircase-owned, no Bayesian posteriors)
        n_dirs = int(config.dir_max - config.dir_min + 1)
        self.dir_hits = np.zeros(n_dirs, dtype=float)
        self.dir_total = np.zeros(n_dirs, dtype=float)

    def _quantize(self, d: float, t: float, direction: int) -> Tuple[float, float, int]:
        d = round(d / self.cfg.quantize_d) * self.cfg.quantize_d
        t = round(t / self.cfg.quantize_t) * self.cfg.quantize_t
        direction = int(round(direction / self.cfg.quantize_dir) * self.cfg.quantize_dir)
        return d, t, direction

    def _rank_dirs(self) -> List[int]:
        """Rank directions by raw hit rate (lowest = hardest). Pure staircase counting."""
        hit_rate = np.where(
            self.dir_total > 0,
            self.dir_hits / self.dir_total,
            0.5,  # unknown directions assumed neutral
        )
        # Lower hit rate => harder. Stable sort so ties keep deterministic order.
        return list(np.argsort(hit_rate, kind="stable"))

    def record_dir_outcome(self, direction: int, hit: bool) -> None:
        """Record hit/miss for a direction (staircase-owned counters)."""
        idx = int(np.clip(direction - self.cfg.dir_min, 0, len(self.dir_hits) - 1))
        self.dir_total[idx] += 1.0
        if hit:
            self.dir_hits[idx] += 1.0

    def _move_dir(self, harder: bool) -> None:
        order = self._rank_dirs()
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
        return str(self.rng.choice(["d", "t", "dir"]))

    def _make_harder(self) -> None:
        adjust_var = self._sample_adjust_var()
        if adjust_var == "d":
            self.d += self.cfg.d_weight * self.cfg.d_step
        elif adjust_var == "t":
            self.t -= self.cfg.t_weight * self.cfg.t_step
        else:
            self._move_dir(harder=True)

        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(self.t, self.cfg.t_min, self.cfg.t_max)
        self.dir = int(clamp(self.dir, self.cfg.dir_min, self.cfg.dir_max))
        self.d, self.t, self.dir = self._quantize(self.d, self.t, self.dir)

    def _make_easier(self) -> None:
        adjust_var = self._sample_adjust_var()
        if adjust_var == "d":
            self.d -= self.cfg.d_weight * self.cfg.d_step
        elif adjust_var == "t":
            self.t += self.cfg.t_weight * self.cfg.t_step
        else:
            self._move_dir(harder=False)

        self.d = clamp(self.d, self.cfg.d_min, self.cfg.d_max)
        self.t = clamp(self.t, self.cfg.t_min, self.cfg.t_max)
        self.dir = int(clamp(self.dir, self.cfg.dir_min, self.cfg.dir_max))
        self.d, self.t, self.dir = self._quantize(self.d, self.t, self.dir)

    def update(self, hit: bool) -> Tuple[float, float, int, str]:
        if hit:
            self.success_streak += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
            self.success_streak = 0

        if self.success_streak >= self.cfg.k_up:
            self._make_harder()
            self.success_streak = 0
            return self.d, self.t, self.dir, "harder"

        if self.fail_streak >= self.cfg.k_down:
            self._make_easier()
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
    dir0: int = 2,
    cfg: StaircaseConfig | None = None,
    calibration: bool = True,
) -> Dict[str, List]:
    rng = np.random.default_rng(seed)

    # --- Calibration and bounds derivation ---
    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)

    if cfg is None:
        # Derive generous bounds from calibration data
        cal_d_min, cal_d_max, cal_t_min, cal_t_max = derive_bounds_from_calibration(
            calibration_result, patient)
        cfg = StaircaseConfig(
            d_min=cal_d_min, d_max=cal_d_max,
            t_min=cal_t_min, t_max=cal_t_max,
        )

    controller = StaircaseController(cfg, d0=d0, t0=t0, dir0=dir0, rng=rng)

    # Seed staircase direction counters from calibration trials
    if calibration_result:
        for trial in calibration_result.get("trials", []):
            direction = int(np.clip(trial.get("direction_bin", 0), 0, 4))
            hit = bool(trial.get("hit", False))
            controller.record_dir_outcome(direction, hit)

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
    observed_speeds = []

    for k in range(n_trials):
        # Backup plan: check bounds expansion every 50 trials
        if k > 0 and k % 50 == 0 and len(observed_speeds) >= 5:
            new_d_min, new_d_max, new_t_min, new_t_max, changed = expand_bounds_if_needed(
                cfg.d_min, cfg.d_max, cfg.t_min, cfg.t_max, observed_speeds)
            if changed:
                cfg.d_min, cfg.d_max = new_d_min, new_d_max
                cfg.t_min, cfg.t_max = new_t_min, new_t_max

        resample = False
        d_sys, t_sys = controller.d, controller.t
        direction = int(controller.dir)
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

        # Record direction outcome in staircase-owned counters
        controller.record_dir_outcome(direction, hit)
        _, _, _, action = controller.update(hit)
        if resample:
            action = f"resample: d,t,dir={controller.d:.2f},{controller.t:.2f},{controller.dir}"

        d_pat = float(outcome["dist_ratio"]) * d_sys
        t_pat = float(outcome["t_pat"])

        # Track observed speed for backup expansion
        if hit and t_pat > 0.01:
            observed_speeds.append(d_pat / t_pat)
        elif t_sys > 0.01:
            observed_speeds.append(d_pat / t_sys)

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
