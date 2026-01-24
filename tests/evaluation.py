import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from staircasing import StaircaseConfig, bin25, run_staircase_sim

# horizontal: distance levels (0-4) (0 = closest, 4 = farthest)
# vertical: time levels (0-4) (0 = quickest, 4 = longest)
PATIENT_PROFILE = {
    "slow-close":np.array([
        [12, 12,  8,  4,  0],
        [10, 12, 12,  8,  4],
        [ 8, 10, 12, 12,  8],
        [ 4,  8, 10, 12, 12],
        [ 0,  4,  8, 10, 12],
    ],dtype=float),
    "fast-far":np.array([
        [12, 12,  8,  4,  0],
        [10, 12, 12,  8,  4],
        [ 8, 10, 12, 12,  8],
        [ 4,  8, 10, 12, 12],
        [ 0,  4,  8, 10, 12]
    ],dtype=float),
    "slow-far":np.array([
        [10, 12, 12, 12, 12],
        [ 8, 10, 12, 12, 12],
        [ 4,  8, 10, 12, 12],
        [ 0,  4,  8, 10, 12],
        [ 0,  0,  4,  8, 10],
    ],dtype=float),
    "fast-close":np.array([
        [ 8,  8,  4,  0,  0],
        [10, 10,  8,  4,  0],
        [12, 12, 10,  8,  4],
        [12, 12, 12, 10,  8],
        [12, 12, 12, 12, 10],
    ],dtype=float),
}

_BASE_PROFILE = np.array([
    [12, 8, 4, 0, 0],
    [8, 12, 8, 4, 0],
    [4, 8, 12, 8, 4],
    [0, 4, 8, 12, 8],
    [0, 0, 4, 8, 12],
], dtype=float)

def desired_counts_5x5(total: int = 120, patient_profile: Optional[str] = "slow-close") -> np.ndarray:
    # Base template (sums to 116): peak at center, taper to edges
    if patient_profile is None:
        base = _BASE_PROFILE
    else:
        base = PATIENT_PROFILE[patient_profile]

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
    cfg: StaircaseConfig,
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

    miss_idx = np.where(hit == 0)[0]
    ax.scatter(trials[miss_idx], d[miss_idx], s=12, marker="x", label="miss (on d)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    ax.set_title("Staircase trajectory of (d, t) over trials")
    plt.tight_layout()
    plt.show()


def plot_hit_rate(logs: Dict[str, List], window: int = 15) -> None:
    hit = np.array(logs["hit"], dtype=float)
    trials = np.array(logs["trial"])

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
    cfg: StaircaseConfig,
) -> None:
    d = logs["d_sys"]
    t = logs["t_sys"]

    achieved = np.zeros((5, 5), dtype=int)
    for di, ti in zip(d, t):
        i, j = bin25(di, ti, cfg)
        achieved[i, j] += 1

    diff = achieved - target

    for title, mat in [
        ("Achieved trial counts per (d,t) bin", achieved),
        ("Desired trial counts per (d,t) bin", target),
        ("Achieved - Desired (positive means overfilled)", diff),
    ]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(mat, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Time bin (0=shortest ... 4=longest)")
        ax.set_ylabel("Distance bin (0=closest ... 4=farthest)")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center")

        plt.tight_layout()
        plt.show()


def plot_coverage_over_time(
    logs: Dict[str, List],
    target: np.ndarray,
    cfg: StaircaseConfig,
) -> None:
    d = logs["d_sys"]
    t = logs["t_sys"]
    satisfied = []
    counts = np.zeros((5, 5), dtype=int)

    for k, (di, ti) in enumerate(zip(d, t), start=1):
        i, j = bin25(di, ti, cfg)
        counts[i, j] += 1
        satisfied.append(int((counts >= target).sum()))

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

    target = desired_counts_5x5(total=target_total, patient_profile=None)

    times: List[Optional[int]] = []
    first_logs = None
    for idx, s in enumerate(seeds):
        logs = run_staircase_sim(n_trials=n_trials, seed=s, d0=d0, t0=t0, cfg=cfg)
        t_hit = trials_to_meet_target_bins(logs["d_sys"], logs["t_sys"], target, cfg)
        times.append(t_hit)
        if idx == 0:
            first_logs = logs

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
    cfg = StaircaseConfig(
        d_min=0.10, d_max=0.80,
        t_min=1.0, t_max=7.0,
        d_step=0.1, t_step=0.5,
        k_up=1, k_down=1,
        couple_distance_and_time=True,
        quantize_d=0.01, quantize_t=0.01,
    )

    summary = evaluate_staircase_vs_targets(
        n_trials=240,
        seeds=[7, 8, 9, 10, 11],
        cfg=cfg,
        target_total=120,
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
