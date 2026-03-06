from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_shared_patient_profiles():
    module = _load_module_from_path(
        "patient_profiles_shared",
        BASE_DIR / "patient_profiles_shared.py",
    )
    return {k: dict(v) for k, v in module.PATIENT_PROFILES.items()}


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel
VIS_PATH = BASE_DIR / "tests" / "visualization.py"
viz_mod = _load_module_from_path("tests.visualization", VIS_PATH)

D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0
N_DIRECTIONS = 5

# Canonicalize profiles across 0/1/2/3 var panels.
PATIENT_PROFILES = _load_shared_patient_profiles()
# Keep random distance and direction, but use one fixed t per profile for 0-var sweeps.
PATIENT_FIXED_T = {
    "overall_weak": 4.0,
    "overall_medium": 4.0,
    "overall_strong": 4.0,
    "highspeed_lowrom": 4.0,
    "lowspeed_highrom": 4.0,
}


def level5(x: float, xmin: float, xmax: float) -> int:
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


def bin25(d: float, t: float) -> tuple[int, int]:
    return level5(d, D_MIN, D_MAX), level5(t, T_MIN, T_MAX)


def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    d_means = np.asarray(patient.d_means, dtype=float)
    idx = np.where(d_means <= d_sys)[0]
    return int(idx[-1]) if len(idx) else 0


def rolling_mean(values: list[float], window: int = 50) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    out = np.empty_like(arr, dtype=float)
    csum = np.cumsum(arr, dtype=float)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def run_random_sim(
    patient: PatientModel,
    n_trials: int = 500,
    seed: int = 7,
    patient_profile: str | None = None,
    t_fixed: float | None = None,
) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.zeros((5, 5), dtype=int)

    logs = {
        "trial": [],
        "d": [],
        "t": [],
        "direction": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "t_pat": [],
        "d_pat": [],
    }

    previous_hit = True
    fixed_t = None if t_fixed is None else float(t_fixed)
    if fixed_t is None and patient_profile in PATIENT_FIXED_T:
        fixed_t = float(PATIENT_FIXED_T[patient_profile])

    for k in range(n_trials):
        d_sys = float(rng.uniform(D_MIN, D_MAX))
        t_sys = fixed_t if fixed_t is not None else float(rng.uniform(T_MIN, T_MAX))
        direction = int(rng.integers(0, N_DIRECTIONS))

        lvl = distance_level_from_patient_bins(patient, d_sys)
        out = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
            direction_bin=direction,
        )

        hit = int(bool(out["hit"]))
        previous_hit = bool(hit)

        i, j = bin25(d_sys, t_sys)
        counts[i, j] += 1

        logs["trial"].append(k)
        logs["d"].append(d_sys)
        logs["t"].append(t_sys)
        logs["direction"].append(direction)
        logs["hit"].append(hit)
        logs["time_ratio"].append(float(out["time_ratio"]))
        logs["dist_ratio"].append(float(out["dist_ratio"]))
        logs["t_pat"].append(float(out["t_pat"]))
        logs["d_pat"].append(float(out["d_pat"]))

    return logs, counts


def plot_outputs(logs: dict, counts: np.ndarray, assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)

    xlabels = ["shortest", "short", "medium", "long", "longest"]
    ylabels = ["closest", "close", "medium", "far", "farthest"]

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(counts, aspect="auto")
    ax1.set_title("0-var Random Counts Heatmap")
    ax1.set_xticks(range(len(xlabels)))
    ax1.set_xticklabels(xlabels)
    ax1.set_yticks(range(len(ylabels)))
    ax1.set_yticklabels(ylabels)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax1.text(j, i, str(int(counts[i, j])), ha="center", va="center", fontsize=8)
    fig1.colorbar(im, ax=ax1)
    fig1.tight_layout()
    fig1.savefig(assets_dir / "random_counts_heatmap.png", dpi=150)
    plt.close(fig1)

    trials = np.asarray(logs["trial"], dtype=int)
    hits = np.asarray(logs["hit"], dtype=float)
    rolling_hit = rolling_mean(hits.tolist(), window=50)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(trials, rolling_hit, label="rolling hit (50)")
    ax2.axhline(float(np.mean(hits)), linestyle=":", label=f"overall={np.mean(hits):.3f}")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("0-var Rolling Hit Rate")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Hit Rate")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(assets_dir / "random_rolling_hit_rate.png", dpi=150)
    plt.close(fig2)

    dirs = np.asarray(logs["direction"], dtype=int)
    dir_counts = np.zeros(N_DIRECTIONS, dtype=int)
    for d in dirs:
        if 0 <= d < N_DIRECTIONS:
            dir_counts[d] += 1

    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    ax3.bar(np.arange(N_DIRECTIONS), dir_counts)
    ax3.set_xticks(np.arange(N_DIRECTIONS))
    ax3.set_title("0-var Direction Usage")
    ax3.set_xlabel("Direction Bin")
    ax3.set_ylabel("Count")
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(assets_dir / "random_direction_usage.png", dpi=150)
    plt.close(fig3)


def plot_caterpillars_random_by_profile(
    assets_dir: Path,
    n_trials: int = 500,
    seed: int = 7,
) -> None:
    profile_stats = {}
    all_t_values = []
    all_d_values = []
    all_dir_values = []
    algorithm_name = "random_0var"

    for profile_name, params in PATIENT_PROFILES.items():
        patient = PatientModel(**params)
        logs, _counts = run_random_sim(
            patient=patient,
            n_trials=n_trials,
            seed=seed,
            patient_profile=profile_name,
        )

        mean_t, std_t = viz_mod.average_time(logs)
        mean_d, std_d = viz_mod.average_distance(logs)
        mean_dir, std_dir = viz_mod.average_direction(logs)

        profile_stats[profile_name] = {
            "algorithm_names": [algorithm_name],
            "means_time": [mean_t],
            "std_time": [std_t],
            "means_dist": [mean_d],
            "std_dist": [std_d],
            "means_dir": [mean_dir],
            "std_dir": [std_dir],
        }

        all_t_values.extend(np.asarray(logs.get("t", []), dtype=float).tolist())
        all_d_values.extend(np.asarray(logs.get("d", []), dtype=float).tolist())
        all_dir_values.extend(
            [viz_mod.DIR_INDEX_TO_ANGLE.get(int(d), float(d)) for d in logs.get("direction", [])]
        )

    time_xlim = (
        float(np.min(all_t_values)),
        float(np.max(all_t_values)),
    ) if all_t_values else None
    dist_xlim = (
        float(np.min(all_d_values)),
        float(np.max(all_d_values)),
    ) if all_d_values else None
    dir_xlim = (
        float(np.min(all_dir_values)),
        float(np.max(all_dir_values)),
    ) if all_dir_values else None

    viz_mod.plot_caterpillar_means_by_algorithm(
        profile_stats=profile_stats,
        title="0-var Random: Mean/SD by Patient Profile",
        time_xlim=time_xlim,
        dist_xlim=dist_xlim,
        dir_xlim=dir_xlim,
        save_path=assets_dir / "random_caterpillar_by_algorithm.png",
        show=False,
    )
    viz_mod.plot_caterpillar_means_by_profile(
        profile_stats=profile_stats,
        title="0-var Random: Mean/SD by Patient Profile",
        time_xlim=time_xlim,
        dist_xlim=dist_xlim,
        dir_xlim=dir_xlim,
        save_path=assets_dir / "random_caterpillar_by_profile.png",
        show=False,
    )


def plot_rolling_hit_rate_by_profile(
    assets_dir: Path,
    n_trials: int = 500,
    seed: int = 7,
    window: int = 50,
) -> None:
    hits_by_profile = {}
    for profile_name, params in PATIENT_PROFILES.items():
        patient = PatientModel(**params)
        logs, _counts = run_random_sim(
            patient=patient,
            n_trials=n_trials,
            seed=seed,
            patient_profile=profile_name,
        )
        hits_by_profile[profile_name] = logs.get("hit", [])

    viz_mod.plot_rolling_hit_rate(
        hit_series_by_algorithm=hits_by_profile,
        window=window,
        min_periods=1,
        title="0-var Random Rolling Hit Rate by Patient Profile",
        save_path=assets_dir / "random_rolling_hit_rate_by_profile.png",
        show=False,
    )


def main() -> None:
    patient = PatientModel(seed=7)
    logs, counts = run_random_sim(
        patient=patient,
        n_trials=500,
        seed=7,
        patient_profile="overall_medium",
    )

    assets_dir = BASE_DIR / "Assets" / "0_var"
    plot_outputs(logs, counts, assets_dir)
    plot_caterpillars_random_by_profile(assets_dir=assets_dir, n_trials=500, seed=7)
    plot_rolling_hit_rate_by_profile(assets_dir=assets_dir, n_trials=500, seed=7, window=50)

    hit_rate = float(np.mean(np.asarray(logs["hit"], dtype=float))) if logs["hit"] else float("nan")
    mean_t = float(np.mean(np.asarray(logs["t"], dtype=float))) if logs["t"] else float("nan")
    mean_d = float(np.mean(np.asarray(logs["d"], dtype=float))) if logs["d"] else float("nan")

    print(f"Trials: {len(logs['trial'])}")
    print(f"Mean hit rate: {hit_rate:.3f}")
    print(f"Mean t: {mean_t:.3f} s")
    print(f"Mean d: {mean_d:.3f} m")
    print(f"Saved plots to: {assets_dir}")


if __name__ == "__main__":
    main()
