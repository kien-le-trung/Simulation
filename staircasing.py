from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import random

# ---- Import your patient simulator (patient_simulation_v2.py) ----
from patient_simulation_v2 import PatientModel


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


# ============================================================
# Desired counts design (variability + just-hard-enough)
# ============================================================

def desired_counts_5x5(total: int = 120) -> np.ndarray:
    # Base template (sums to 116): peak at center, taper to edges
    base = np.array([
        [12, 8, 4, 0, 0],
        [8, 12, 8, 4, 0],
        [4, 8, 12, 8, 4],
        [0, 4, 8, 12, 8],
        [0, 0, 4, 8, 12],
    ], dtype=float)

    # Scale to desired total
    s = base.sum()
    if total <= 0:
        raise ValueError("total must be > 0")
    scaled = base * (total / s)

    # Round while keeping exact sum via largest-remainder method
    flo = np.floor(scaled).astype(int)
    remainder = scaled - flo
    missing = total - flo.sum()
    if missing > 0:
        flat_idx = np.argsort(remainder.ravel())[::-1]  # descending remainders
        for k in range(missing):
            i = flat_idx[k] // 5
            j = flat_idx[k] % 5
            flo[i, j] += 1
    return flo


def trials_to_meet_target_bins(
    d_list: List[float],
    t_list: List[float],
    target: np.ndarray,
    cfg: StaircaseConfig
) -> Optional[int]:
    """
    Iterate through trials and return the first trial index (1-based count)
    where cumulative bin counts >= target in ALL bins.
    If never met, return None.
    """
    counts = np.zeros((5, 5), dtype=int)
    for k, (d, t) in enumerate(zip(d_list, t_list), start=1):
        i, j = bin25(d, t, cfg)
        counts[i, j] += 1
        if np.all(counts >= target):
            return k
    return None


# ============================================================
# Simulation
# ============================================================

def run_staircase_sim(
    n_trials: int = 150,
    seed: int = 7,
    d0: float = 0.30,
    t0: float = 4.0,
    cfg: StaircaseConfig | None = None,
) -> Dict[str, List]:
    cfg = cfg or StaircaseConfig()
    patient = PatientModel(seed=seed)
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

    return logs


# ============================================================
# Visualization helpers (matplotlib only)
# ============================================================
def plot_trajectory(logs: Dict[str, List], cfg: StaircaseConfig) -> None:
    trials = np.array(logs["trial"])
    d = np.array(logs["d_sys"])
    t = np.array(logs["t_sys"])
    hit = np.array(logs["hit"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trials, d, label="d_sys (m)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Distance d (m)")
    ax.set_ylim(cfg.d_min - 0.02, cfg.d_max + 0.02)

    ax2 = ax.twinx()
    ax2.plot(trials, t, label="t_sys (s)", linestyle="--")
    ax2.set_ylabel("Time t (s)")
    ax2.set_ylim(cfg.t_min - 0.2, cfg.t_max + 0.2)

    # Mark hits/misses lightly
    miss_idx = np.where(hit == 0)[0]
    ax.scatter(trials[miss_idx], d[miss_idx], s=12, marker="x", label="miss (on d)")

    # Legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    ax.set_title("Staircase trajectory of (d, t) over trials")
    plt.tight_layout()
    plt.show()


def plot_hit_rate(logs: Dict[str, List], window: int = 15) -> None:
    hit = np.array(logs["hit"], dtype=float)
    trials = np.array(logs["trial"])

    # Rolling mean
    roll = np.full_like(hit, np.nan)
    for k in range(len(hit)):
        lo = max(0, k - window + 1)
        roll[k] = hit[lo:k + 1].mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(trials, roll, label=f"rolling hit rate (window={window})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Hit rate")
    ax.set_title("Rolling hit rate")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_bin_heatmaps(
    logs: Dict[str, List],
    target: np.ndarray,
    cfg: StaircaseConfig
) -> None:
    d = logs["d_sys"]
    t = logs["t_sys"]

    achieved = np.zeros((5, 5), dtype=int)
    for di, ti in zip(d, t):
        i, j = bin25(di, ti, cfg)
        achieved[i, j] += 1

    diff = achieved - target

    # Plot achieved, target, diff
    for title, mat in [
        ("Achieved trial counts per (d,t) bin", achieved),
        ("Desired trial counts per (d,t) bin", target),
        ("Achieved - Desired (positive means overfilled)", diff),
    ]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(mat, origin="lower")  # origin lower: bin 0 at bottom
        ax.set_title(title)
        ax.set_xlabel("Time bin (0=shortest ... 4=longest)")
        ax.set_ylabel("Distance bin (0=closest ... 4=farthest)")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # annotate cells
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center")

        plt.tight_layout()
        plt.show()


def plot_coverage_over_time(
    logs: Dict[str, List],
    target: np.ndarray,
    cfg: StaircaseConfig
) -> None:
    d = logs["d_sys"]
    t = logs["t_sys"]
    satisfied = []
    counts = np.zeros((5, 5), dtype=int)

    for k, (di, ti) in enumerate(zip(d, t), start=1):
        i, j = bin25(di, ti, cfg)
        counts[i, j] += 1
        satisfied.append(int((counts >= target).sum()))  # number of satisfied cells (0..25)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(satisfied) + 1), satisfied, label="# bins satisfied")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Satisfied bins (out of 25)")
    ax.set_title("Progress toward desired bin-coverage targets")
    ax.set_ylim(-1, 26)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Evaluation runner
# ============================================================

def evaluate_staircase_vs_targets(
    n_trials: int = 200,
    seeds: List[int] | None = None,
    cfg: StaircaseConfig | None = None,
    target_total: int = 120,
    d0: float = 0.30,
    t0: float = 4.0,
    make_plots_for_first_seed: bool = True,
) -> Dict[str, object]:
    """
    Runs multiple seeds and reports:
      - time_to_target: trials needed until ALL bins meet desired counts (or None)
      - achieved counts summary for the first seed
      - plots for first seed (optional)
    """
    cfg = cfg or StaircaseConfig()
    seeds = seeds or [1, 2, 3, 4, 5]

    target = desired_counts_5x5(total=target_total)

    times: List[Optional[int]] = []
    first_logs = None
    for idx, s in enumerate(seeds):
        logs = run_staircase_sim(n_trials=n_trials, seed=s, d0=d0, t0=t0, cfg=cfg)
        t_hit = trials_to_meet_target_bins(logs["d_sys"], logs["t_sys"], target, cfg)
        times.append(t_hit)
        if idx == 0:
            first_logs = logs

    # summarize
    met = [x for x in times if x is not None]
    summary = {
        "seeds": seeds,
        "n_trials_run": n_trials,
        "target_total": int(target.sum()),
        "time_to_target_each_seed": times,
        "num_met_target": len(met),
        "num_not_met_target": len(times) - len(met),
        "mean_time_to_target_if_met": float(np.mean(met)) if met else None,
        "median_time_to_target_if_met": float(np.median(met)) if met else None,
        "target_matrix_5x5": target,
    }

    if make_plots_for_first_seed and first_logs is not None:
        plot_trajectory(first_logs, cfg)
        plot_hit_rate(first_logs, window=15)
        plot_bin_heatmaps(first_logs, target, cfg)
        plot_coverage_over_time(first_logs, target, cfg)

    return summary


if __name__ == "__main__":
    # Example config: 2-up 1-down staircase coupled in (d,t)
    cfg = StaircaseConfig(
        d_min=0.10, d_max=0.80,
        t_min=1.0, t_max=7.0,
        d_step=0.1, t_step=0.5,
        k_up=1, k_down=1,
        couple_distance_and_time=True,
        quantize_d=0.01, quantize_t=0.01,
    )

    summary = evaluate_staircase_vs_targets(
        n_trials=240,               # give it enough budget to try to fill targets
        seeds=[7, 8, 9, 10, 11],
        cfg=cfg,
        target_total=120,           # desired coverage amount (spread across 25 bins)
        d0=0.30,
        t0=4.0,
        make_plots_for_first_seed=True,
    )

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        if k == "target_matrix_5x5":
            continue
        print(f"{k}: {v}")

    print("\nDesired target matrix (distance bins rows 0..4, time bins cols 0..4):")
    print(summary["target_matrix_5x5"])