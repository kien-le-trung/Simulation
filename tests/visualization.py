import matplotlib.pyplot as plt
import numpy as np

# heatmaps
xlabels = ["shortest", "short", "medium", "long", "longest"]  # time bins
ylabels = ["closest", "close", "medium", "far", "farthest"]   # distance bins
N_DIRECTIONS = 9
N_AZIMUTH_BINS = 3
N_ELEVATION_BINS = 3
AZIMUTH_RANGE_DEG = 110.0
ELEVATION_RANGE_DEG = 96.0


def _bin_center_angle(bin_idx: int, n_bins: int, total_range_deg: float) -> float:
    bin_width = total_range_deg / float(n_bins)
    min_angle = -0.5 * total_range_deg
    return min_angle + (float(bin_idx) + 0.5) * bin_width


def direction_index_to_bins(direction: int) -> tuple[int, int]:
    idx = int(np.clip(direction, 0, N_DIRECTIONS - 1))
    az_idx = idx % N_AZIMUTH_BINS
    el_idx = idx // N_AZIMUTH_BINS
    return az_idx, el_idx


def direction_index_to_azimuth(direction: int) -> float:
    az_idx, _ = direction_index_to_bins(direction)
    return _bin_center_angle(az_idx, N_AZIMUTH_BINS, AZIMUTH_RANGE_DEG)


def direction_index_to_elevation(direction: int) -> float:
    _, el_idx = direction_index_to_bins(direction)
    return _bin_center_angle(el_idx, N_ELEVATION_BINS, ELEVATION_RANGE_DEG)


DIR_INDEX_TO_AZIMUTH_ANGLE = {
    d: direction_index_to_azimuth(d) for d in range(N_DIRECTIONS)
}
DIR_INDEX_TO_ELEVATION_ANGLE = {
    d: direction_index_to_elevation(d) for d in range(N_DIRECTIONS)
}
# Backward-compatible alias used by existing callers.
DIR_INDEX_TO_ANGLE = DIR_INDEX_TO_AZIMUTH_ANGLE


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
        d_vals = np.asarray(model.d_means, dtype=float)
        d_vals = np.concatenate(([0.0], d_vals))
        v_mean = model._mean_speed(d_vals)
        ax.plot(d_vals, v_mean, label=name, linewidth=2.0)

    ax.set_title("Mean speed against time")
    ax.set_xlabel("Distance reached (m)")
    ax.set_ylabel("Mean speed of the patient (m/s)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
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
    n_dir=9,
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
    labels = []
    for d in range(N_DIRECTIONS):
        az = DIR_INDEX_TO_AZIMUTH_ANGLE.get(d, float(d))
        el = DIR_INDEX_TO_ELEVATION_ANGLE.get(d, float(d))
        labels.append(f"az {az:.1f}, el {el:.1f}")
    ax.set_xticklabels(labels, rotation=45, ha="right")
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
    angles = np.array(
        [DIR_INDEX_TO_AZIMUTH_ANGLE.get(int(d), d) for d in directions],
        dtype=float,
    )
    return float(np.mean(angles)), float(np.std(angles))


def average_direction_components(hist):
    directions = np.asarray(hist.get("direction", []), dtype=float)
    if directions.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    az_angles = np.array(
        [DIR_INDEX_TO_AZIMUTH_ANGLE.get(int(d), d) for d in directions],
        dtype=float,
    )
    el_angles = np.array(
        [DIR_INDEX_TO_ELEVATION_ANGLE.get(int(d), d) for d in directions],
        dtype=float,
    )
    return (
        float(np.mean(az_angles)),
        float(np.std(az_angles)),
        float(np.mean(el_angles)),
        float(np.std(el_angles)),
    )


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


def rolling_hit_rates(hist, window=10, min_periods=1):
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


def plot_caterpillar_means_by_algorithm(
    profile_stats,
    title="Caterpillar Plot: Mean and Std by Algorithm",
    time_xlim=None,
    dist_xlim=None,
    dir_xlim=None,
    save_path=None,
    show=True,
):
    """
    Create one figure with a row per algorithm and 3 columns (time/dist/direction).

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
    first_profile = profile_stats[profile_names[0]]
    algorithm_names = list(first_profile["algorithm_names"])
    n_algorithms = len(algorithm_names)
    n_profiles = len(profile_names)

    # Validate that each profile reports stats for the same algorithms in same order.
    for profile_name in profile_names:
        stats = profile_stats[profile_name]
        if list(stats["algorithm_names"]) != algorithm_names:
            raise ValueError("All profiles must have the same algorithm_names order")
        if not (
            len(stats["means_time"]) == len(stats["std_time"]) == n_algorithms
            and len(stats["means_dist"]) == len(stats["std_dist"]) == n_algorithms
            and len(stats["means_dir"]) == len(stats["std_dir"]) == n_algorithms
        ):
            raise ValueError("Each profile must provide one mean/std per algorithm")

    fig, axes = plt.subplots(
        n_algorithms,
        3,
        figsize=(12, max(3.5, 2.6 * n_algorithms)),
        sharex="col",
        squeeze=False,
    )
    xlims = [time_xlim, dist_xlim, dir_xlim]

    for row_idx, algorithm_name in enumerate(algorithm_names):
        y = np.arange(n_profiles)
        means_time = [profile_stats[p]["means_time"][row_idx] for p in profile_names]
        std_time = [profile_stats[p]["std_time"][row_idx] for p in profile_names]
        means_dist = [profile_stats[p]["means_dist"][row_idx] for p in profile_names]
        std_dist = [profile_stats[p]["std_dist"][row_idx] for p in profile_names]
        means_dir = [profile_stats[p]["means_dir"][row_idx] for p in profile_names]
        std_dir = [profile_stats[p]["std_dir"][row_idx] for p in profile_names]

        panels = [
            ("Time", means_time, std_time),
            ("Distance", means_dist, std_dist),
            ("Direction", means_dir, std_dir),
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
        axes[row_idx, 0].set_yticklabels(profile_names)
        axes[row_idx, 0].set_ylabel(algorithm_name)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Distance (m)")
    axes[-1, 2].set_xlabel("Direction (degrees)")

    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_caterpillar_means_by_profile(
    profile_stats,
    title="Caterpillar Plot: Mean and Std by Profile",
    time_xlim=None,
    dist_xlim=None,
    dir_xlim=None,
    dir_az_xlim=None,
    dir_el_xlim=None,
    save_path=None,
    show=True,
):
    if not isinstance(profile_stats, dict) or len(profile_stats) == 0:
        raise ValueError("profile_stats must be a non-empty dict")

    profile_names = list(profile_stats.keys())
    n_profiles = len(profile_names)
    first_stats = profile_stats[profile_names[0]]
    has_components = all(
        k in first_stats for k in ("means_dir_az", "std_dir_az", "means_dir_el", "std_dir_el")
    )
    n_cols = 4 if has_components else 3

    fig, axes = plt.subplots(
        n_profiles,
        n_cols,
        figsize=(4 * n_cols, max(3.5, 2.6 * n_profiles)),
        sharex="col",
        squeeze=False,
    )
    az_xlim = dir_az_xlim if dir_az_xlim is not None else dir_xlim
    xlims = [time_xlim, dist_xlim, az_xlim]
    if has_components:
        xlims.append(dir_el_xlim)

    for row_idx, profile_name in enumerate(profile_names):
        stats = profile_stats[profile_name]
        algorithm_names = stats["algorithm_names"]
        n_algs = len(algorithm_names)
        y = np.arange(n_algs)

        panels = [
            ("Time", stats["means_time"], stats["std_time"]),
            ("Distance", stats["means_dist"], stats["std_dist"]),
        ]
        if has_components:
            panels.extend(
                [
                    ("Azimuth", stats["means_dir_az"], stats["std_dir_az"]),
                    ("Elevation", stats["means_dir_el"], stats["std_dir_el"]),
                ]
            )
        else:
            panels.append(("Direction", stats["means_dir"], stats["std_dir"]))

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
    axes[-1, 2].set_xlabel("Azimuth (degrees)" if has_components else "Direction (degrees)")
    if has_components:
        axes[-1, 3].set_xlabel("Elevation (degrees)")

    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_caterpillar_means_by_algorithm(
    profile_stats,
    title="Caterpillar Plot: Mean and Std by Algorithm",
    time_xlim=None,
    dist_xlim=None,
    dir_xlim=None,
    dir_az_xlim=None,
    dir_el_xlim=None,
    save_path=None,
    show=True,
):
    if not isinstance(profile_stats, dict) or len(profile_stats) == 0:
        raise ValueError("profile_stats must be a non-empty dict")

    profile_names = list(profile_stats.keys())
    first_profile = profile_stats[profile_names[0]]
    algorithm_names = list(first_profile["algorithm_names"])
    n_algorithms = len(algorithm_names)
    n_profiles = len(profile_names)
    has_components = all(
        k in first_profile for k in ("means_dir_az", "std_dir_az", "means_dir_el", "std_dir_el")
    )

    for profile_name in profile_names:
        stats = profile_stats[profile_name]
        if list(stats["algorithm_names"]) != algorithm_names:
            raise ValueError("All profiles must have the same algorithm_names order")
        if not (
            len(stats["means_time"]) == len(stats["std_time"]) == n_algorithms
            and len(stats["means_dist"]) == len(stats["std_dist"]) == n_algorithms
        ):
            raise ValueError("Each profile must provide time/dist mean/std per algorithm")
        if has_components:
            if not (
                len(stats["means_dir_az"]) == len(stats["std_dir_az"]) == n_algorithms
                and len(stats["means_dir_el"]) == len(stats["std_dir_el"]) == n_algorithms
            ):
                raise ValueError("Each profile must provide azimuth/elevation mean/std per algorithm")
        else:
            if not (len(stats["means_dir"]) == len(stats["std_dir"]) == n_algorithms):
                raise ValueError("Each profile must provide one direction mean/std per algorithm")

    n_cols = 4 if has_components else 3
    fig, axes = plt.subplots(
        n_algorithms,
        n_cols,
        figsize=(4 * n_cols, max(3.5, 2.6 * n_algorithms)),
        sharex="col",
        squeeze=False,
    )
    az_xlim = dir_az_xlim if dir_az_xlim is not None else dir_xlim
    xlims = [time_xlim, dist_xlim, az_xlim]
    if has_components:
        xlims.append(dir_el_xlim)

    for row_idx, algorithm_name in enumerate(algorithm_names):
        y = np.arange(n_profiles)
        means_time = [profile_stats[p]["means_time"][row_idx] for p in profile_names]
        std_time = [profile_stats[p]["std_time"][row_idx] for p in profile_names]
        means_dist = [profile_stats[p]["means_dist"][row_idx] for p in profile_names]
        std_dist = [profile_stats[p]["std_dist"][row_idx] for p in profile_names]

        panels = [("Time", means_time, std_time), ("Distance", means_dist, std_dist)]
        if has_components:
            means_dir_az = [profile_stats[p]["means_dir_az"][row_idx] for p in profile_names]
            std_dir_az = [profile_stats[p]["std_dir_az"][row_idx] for p in profile_names]
            means_dir_el = [profile_stats[p]["means_dir_el"][row_idx] for p in profile_names]
            std_dir_el = [profile_stats[p]["std_dir_el"][row_idx] for p in profile_names]
            panels.extend([("Azimuth", means_dir_az, std_dir_az), ("Elevation", means_dir_el, std_dir_el)])
        else:
            means_dir = [profile_stats[p]["means_dir"][row_idx] for p in profile_names]
            std_dir = [profile_stats[p]["std_dir"][row_idx] for p in profile_names]
            panels.append(("Direction", means_dir, std_dir))

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
        axes[row_idx, 0].set_yticklabels(profile_names)
        axes[row_idx, 0].set_ylabel(algorithm_name)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Distance (m)")
    axes[-1, 2].set_xlabel("Azimuth (degrees)" if has_components else "Direction (degrees)")
    if has_components:
        axes[-1, 3].set_xlabel("Elevation (degrees)")

    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
