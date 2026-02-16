import matplotlib.pyplot as plt
import numpy as np

# heatmaps
xlabels = ["shortest", "short", "medium", "long", "longest"]  # time bins
ylabels = ["closest", "close", "medium", "far", "farthest"]   # distance bins
N_DIRECTIONS = 8
DIR_INDEX_TO_ANGLE = {
    0: 0,
    1: 30,
    2: 45,
    3: 60,
    4: 90,
    5: 120,
    6: 135,
    7: 150,
}


def _finalize_figure(fig, save_path=None, show=True):
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_v_mean_by_profile(patient_profiles, PatientModel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, params in patient_profiles.items():
        model = PatientModel(**params)
        d_vals = model.d_means
        v_mean = model._mean_speed(d_vals)
        ax.plot(d_vals, v_mean, label=name, linewidth=2.0)

    ax.set_title("Mean speed against time")
    ax.set_xlabel("Distance reached (m)")
    ax.set_ylabel("Mean speed of the patient (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_d_by_profile(patient_profiles, PatientModel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, params in patient_profiles.items():
        model = PatientModel(**params)
        ax.plot(model.t_levels, model.d_means, label=name, linewidth=2.0)

    ax.set_title("ROM against time")
    ax.set_xlabel("Time given to patient (s)")
    ax.set_ylabel("Patient's furthest reachable distance (m)")
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_heatmap(mat, title, xlabels, ylabels, annotate=True, save_path=None, show=True):
    fig, ax = plt.subplots()
    ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat.dtype == int:
                    txt = str(mat[i, j])
                else:
                    txt = f"{mat[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(ax.images[0], ax=ax)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)

def phit_tensor_from_hist(
    hist,
    d_min=0.10,
    d_max=0.80,
    t_min=1.0,
    t_max=7.0,
    n_d=5,
    n_t=5,
    n_dir=8,
):
    def _level5(x, xmin, xmax):
        u = (x - xmin) / (xmax - xmin + 1e-12)
        if u < 0.2: return 0
        if u < 0.4: return 1
        if u < 0.6: return 2
        if u < 0.8: return 3
        return 4

    hits = np.zeros((n_d, n_t, n_dir), dtype=float)
    total = np.zeros((n_d, n_t, n_dir), dtype=float)

    for d, t, direction, hit in zip(
        hist.get("d", []),
        hist.get("t", []),
        hist.get("direction", []),
        hist.get("hit", []),
    ):
        i = _level5(float(d), d_min, d_max)
        j = _level5(float(t), t_min, t_max)
        k = int(np.clip(direction, 0, n_dir - 1))
        total[i, j, k] += 1.0
        hits[i, j, k] += 1.0 if hit else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        phit = np.divide(hits, total, out=np.zeros_like(hits), where=total > 0)
    return phit


def plot_phit_d_dir(phit_tensor, t_idx=None, reduce="mean", save_path=None, show=True):
    if t_idx is None:
        if reduce == "mean":
            mat = np.mean(phit_tensor, axis=1)
        elif reduce == "min":
            mat = np.min(phit_tensor, axis=1)
        elif reduce == "max":
            mat = np.max(phit_tensor, axis=1)
        else:
            raise ValueError("reduce must be one of: mean, min, max")
        title = f"P(hit) vs (d,dir), t={reduce}"
    else:
        j = int(np.clip(t_idx, 0, 4))
        mat = phit_tensor[:, j, :]
        title = f"P(hit) vs (d,dir), t_idx={j}"

    fig, ax = plt.subplots()
    ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(N_DIRECTIONS))
    ax.set_xticklabels([f"{DIR_INDEX_TO_ANGLE.get(d, d)} deg" for d in range(N_DIRECTIONS)])
    ax.set_yticks(range(5))
    ax.set_yticklabels(ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(ax.images[0], ax=ax)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
def average_time(hist):
    times = np.asarray(hist.get("t", []), dtype=float)
    if times.size == 0:
        return float("nan")
    return float(np.mean(times)), float(np.std(times))


def average_distance(hist):
    distances = np.asarray(hist.get("d", []), dtype=float)
    if distances.size == 0:
        return float("nan")
    return float(np.mean(distances)), float(np.std(distances))


def average_direction(hist):
    directions = np.asarray(hist.get("direction", []), dtype=float)
    if directions.size == 0:
        return float("nan")
    angles = np.array([DIR_INDEX_TO_ANGLE.get(int(d), d) for d in directions], dtype=float)
    return float(np.mean(angles)), float(np.std(angles))


def rolling_hitting_rate(hist, window=50, min_periods=1):
    """
    Rolling average of hit rate over the last `window` trials.
    Returns a 1D numpy array aligned to hist["hit"] length.
    """
    hits = np.asarray(hist.get("hit", []), dtype=float)
    n = hits.size
    if n == 0:
        return np.array([], dtype=float)
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if min_periods <= 0:
        raise ValueError("min_periods must be a positive integer")

    window = int(window)
    min_periods = int(min_periods)

    csum = np.cumsum(hits, dtype=float)
    rates = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < min_periods:
            rates[i] = np.nan
            continue
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        rates[i] = total / count
    return rates


def rolling_hit_rates(hist, window=50, min_periods=1):
    rolling = rolling_hitting_rate(hist, window=window, min_periods=min_periods)
    hits = np.asarray(hist.get("hit", []), dtype=float)
    overall = float(np.mean(hits)) if hits.size > 0 else float("nan")
    return rolling, overall


def plot_rolling_hit_rate(
    hit_series_by_algorithm,
    window=50,
    min_periods=1,
    title="Rolling Hit Rate",
    save_path=None,
    show=True,
):
    """
    Plot rolling hit rates for multiple algorithms on the same graph.
    Input: dict of {"algorithm_name": hits_list_or_array}
    """
    if not isinstance(hit_series_by_algorithm, dict) or len(hit_series_by_algorithm) == 0:
        raise ValueError("hit_series_by_algorithm must be a non-empty dict")

    fig, ax = plt.subplots()
    plotted_any = False
    for name, hits in hit_series_by_algorithm.items():
        hist = {"hit": hits}
        rolling = rolling_hitting_rate(hist, window=window, min_periods=min_periods)
        if rolling.size == 0:
            continue
        ax.plot(rolling, linewidth=2.0, label=str(name))
        plotted_any = True

    if not plotted_any:
        raise ValueError("No non-empty hit series to plot")

    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Rolling hit rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
def plot_caterpillar_means(
    algorithm_names,
    means_time,
    means_dist,
    means_dir,
    std_time,
    std_dist,
    std_dir,
    title="Caterpillar Plot: Mean and Std",
    time_xlim=None,
    dist_xlim=None,
    dir_xlim=None,
    save_path=None,
    show=True,
):
    """
    Create a caterpillar plot for mean time, distance, and direction with std whiskers.
    All list inputs should have the same length (one entry per algorithm).
    """
    n = len(algorithm_names)
    if not (len(means_time) == len(means_dist) == len(means_dir) == n):
        raise ValueError("Mean lists must match length of algorithm_names")
    if not (len(std_time) == len(std_dist) == len(std_dir) == n):
        raise ValueError("Std lists must match length of algorithm_names")

    algs = np.asarray(algorithm_names)
    y = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(12, max(3.5, 0.5 * n)), sharey=True)
    panels = [
        ("Time", means_time, std_time),
        ("Distance", means_dist, std_dist),
        ("Direction", means_dir, std_dir),
    ]

    xlims = [time_xlim, dist_xlim, dir_xlim]
    for ax, (label, means, stds), xlim in zip(axes, panels, xlims):
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        ax.errorbar(means, y, xerr=stds, fmt="o", capsize=3, linewidth=1.5)
        ax.set_title(label)
        ax.grid(True, axis="x", alpha=0.3)
        if xlim is not None:
            ax.set_xlim(*xlim)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(algs)
    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_caterpillar_means_by_profile(
    profile_stats,
    title="Caterpillar Plot: Mean and Std by Profile",
    time_xlim=None,
    dist_xlim=None,
    dir_xlim=None,
    save_path=None,
    show=True,
):
    """
    Create one figure with a row per patient profile and 3 columns (time/dist/direction).

    profile_stats format:
      {
        "profile_name": {
          "algorithm_names": [...],
          "means_time": [...], "std_time": [...],
          "means_dist": [...], "std_dist": [...],
          "means_dir": [...],  "std_dir": [...],
        },
        ...
      }
    """
    if not isinstance(profile_stats, dict) or len(profile_stats) == 0:
        raise ValueError("profile_stats must be a non-empty dict")

    profile_names = list(profile_stats.keys())
    n_profiles = len(profile_names)

    fig, axes = plt.subplots(
        n_profiles,
        3,
        figsize=(12, max(3.5, 2.6 * n_profiles)),
        sharex="col",
        squeeze=False,
    )
    xlims = [time_xlim, dist_xlim, dir_xlim]

    for row_idx, profile_name in enumerate(profile_names):
        stats = profile_stats[profile_name]
        algorithm_names = stats["algorithm_names"]
        n_algs = len(algorithm_names)
        y = np.arange(n_algs)

        panels = [
            ("Time", stats["means_time"], stats["std_time"]),
            ("Distance", stats["means_dist"], stats["std_dist"]),
            ("Direction", stats["means_dir"], stats["std_dir"]),
        ]

        for col_idx, ((label, means, stds), xlim) in enumerate(zip(panels, xlims)):
            ax = axes[row_idx, col_idx]
            means = np.asarray(means, dtype=float)
            stds = np.asarray(stds, dtype=float)
            ax.errorbar(means, y, xerr=stds, fmt="o", capsize=3, linewidth=1.5)
            ax.grid(True, axis="x", alpha=0.3)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if row_idx == 0:
                ax.set_title(label)
            if col_idx > 0:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        axes[row_idx, 0].set_yticks(y)
        axes[row_idx, 0].set_yticklabels(algorithm_names)
        axes[row_idx, 0].set_ylabel(profile_name)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Distance (m)")
    axes[-1, 2].set_xlabel("Direction (degrees)")

    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
