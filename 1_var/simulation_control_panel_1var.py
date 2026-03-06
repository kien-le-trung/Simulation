import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

# 1-var simulation control panel for distance-only optimization algorithms.
# Time is fixed per patient profile (borrowed from 0_var); distance is adapted.

PATIENT_PROFILES = {
    "overall_weak": {
        "k_d0_per_sec": 0.15,
        "k_d_decay": 1.0,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.04,
        "max_reach": 0.40,
        "handedness": 0.8,
        "k_dir_decay": 2.0,
    },
    "overall_medium": {
        "k_d0_per_sec": 0.7,
        "k_d_decay": 0.55,
        "v_sigma0": 0.10,
        "v_sigma_growth": 0.024,
        "max_reach": 0.70,
        "handedness": 0.2,
        "k_dir_decay": 0.5,
    },
    "overall_strong": {
        "k_d0_per_sec": 2.0,
        "k_d_decay": 0.1,
        "v_sigma0": 0.20,
        "v_sigma_growth": 0.008,
        "max_reach": 1.0,
        "handedness": 0.0,
        "k_dir_decay": 0.1,
    },
    "highspeed_lowrom": {
        "k_d0_per_sec": 1.5,
        "k_d_decay": 0.1,
        "v_sigma0": 0.12,
        "v_sigma_growth": 0.01,
        "max_reach": 0.50,
        "handedness": 0.8,
        "k_dir_decay": 2.0,
    },
    "lowspeed_highrom": {
        "k_d0_per_sec": 0.2,
        "k_d_decay": 0.1,
        "v_sigma0": 0.06,
        "v_sigma_growth": 0.04,
        "max_reach": 1.0,
        "handedness": 0.0,
        "k_dir_decay": 0.1,
    },
}

T_RANDOM_MIN = 0.3
T_RANDOM_MAX = 7.0
PATIENT_FIXED_T = {
    "overall_weak": 4.0,
    "overall_medium": 4.0,
    "overall_strong": 4.0,
    "highspeed_lowrom": 4.0,
    "lowspeed_highrom": 4.0,
}


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_shared_patient_profiles():
    module = _load_module_from_path(
        "patient_profiles_shared",
        BASE_DIR / "patient_profiles_shared.py",
    )
    return {k: dict(v) for k, v in module.PATIENT_PROFILES.items()}


# Canonicalize profiles across 0/1/2/3 var panels.
PATIENT_PROFILES = _load_shared_patient_profiles()


def _load_patient_model():
    module_path = BASE_DIR / "patients" / "patient_simulation_v4.py"
    module = _load_module_from_path("patients.patient_simulation_v4", module_path)
    return module.PatientModel


def _load_ideal_distribution_module():
    module_path = BASE_DIR / "tests" / "make_ideal_distribution.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    keep_names = {
        "D_MIN",
        "D_MAX",
        "T_MIN",
        "T_MAX",
        "D_LEVELS",
        "T_LEVELS",
        "D_BINS",
        "T_BINS",
        "ACTIONS",
    }
    keep_funcs = {
        "action_to_dt",
        "distance_level_from_patient_bins",
        "estimate_true_phit_matrix",
        "make_ideal_distribution",
    }

    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
                elif isinstance(target, ast.Tuple):
                    targets.extend([elt.id for elt in target.elts if isinstance(elt, ast.Name)])
            if any(name in keep_names for name in targets):
                new_body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep_funcs:
            new_body.append(node)

    mod = ast.Module(body=new_body, type_ignores=[])
    compiled = compile(mod, str(module_path), "exec")
    module_dict = {"np": np, "PatientModel": _load_patient_model()}
    exec(compiled, module_dict)
    return module_dict


def _load_algorithm(algorithm_name):
    algorithms_dir = BASE_DIR / "1_var"
    path = algorithms_dir / f"{algorithm_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"No algorithm file found: {path}")
    module_name = f"one_var.{path.stem}"
    return _load_module_from_path(module_name, path)


def _load_visualization_module():
    module_path = BASE_DIR / "tests" / "visualization.py"
    return _load_module_from_path("tests.visualization", module_path)


def _discretized_d_counts_from_logs(logs, d_min, d_max, n_levels=5):
    d_values = np.asarray(logs.get("d", []), dtype=float)
    if d_values.size == 0:
        return np.zeros(n_levels, dtype=int)
    edges = np.linspace(float(d_min), float(d_max), n_levels + 1)
    clipped = np.clip(d_values, float(d_min), float(d_max))
    counts, _ = np.histogram(clipped, bins=edges)
    return counts.astype(int)


def plot_1var_d_level_counts(
    d_counts,
    title,
    save_path=None,
    show=False,
    level_labels=None,
):
    counts = np.asarray(d_counts, dtype=int).reshape(-1)
    if level_labels is None:
        level_labels = ["closest", "close", "medium", "far", "farthest"]
    if len(level_labels) != len(counts):
        raise ValueError("level_labels length must match number of d levels")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts, color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Distance Level")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.grid(True, axis="y", alpha=0.3)
    ymax = int(np.max(counts)) if counts.size else 0
    ax.set_ylim(0, max(1, ymax + max(1, int(0.08 * max(1, ymax)))))
    for rect, val in zip(bars, counts):
        ax.text(
            rect.get_x() + rect.get_width() * 0.5,
            rect.get_height(),
            str(int(val)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_phit_and_ideal_by_profile(
    profile_name,
    mc_per_cell=300,
    target_prob=0.6,
    variability=0.15,
    total_trials=200,
):
    patient_params = PATIENT_PROFILES[profile_name]
    patient_seed = int(patient_params.get("seed", 7))
    PatientModel = _load_patient_model()

    ideal_mod = _load_ideal_distribution_module()
    estimate_true_phit_matrix = ideal_mod["estimate_true_phit_matrix"]
    make_ideal_distribution = ideal_mod["make_ideal_distribution"]

    class _ProfilePatientModel(PatientModel):
        _profile_params = patient_params

        def __init__(self, *args, **kwargs):
            params = dict(self._profile_params)
            params.setdefault("seed", kwargs.get("seed", 7))
            if "v_sigma0" not in params and "v_sigma" in kwargs:
                params["v_sigma0"] = kwargs["v_sigma"]
            params.update({k: v for k, v in kwargs.items() if k in {"seed"}})
            super().__init__(**params)

    estimate_true_phit_matrix.__globals__["PatientModel"] = _ProfilePatientModel
    estimate_true_phit_matrix.__globals__["np"] = np
    estimate_true_phit_matrix.__globals__["distance_level_from_patient_bins"] = ideal_mod["distance_level_from_patient_bins"]
    estimate_true_phit_matrix.__globals__["D_BINS"] = ideal_mod["D_BINS"]
    estimate_true_phit_matrix.__globals__["T_BINS"] = ideal_mod["T_BINS"]

    phit_true = estimate_true_phit_matrix(
        patient_seed=patient_seed,
        patient_speed=patient_params.get("k_d0_per_sec", 0.15),
        patient_speed_sd=patient_params.get("v_sigma0", 0.04),
        k_d_decay=patient_params.get("k_d_decay", 0),
        v_sigma_growth=patient_params.get("v_sigma_growth", 0.03),
        spatial_strength_map=patient_params.get("spatial_strength_map", None),
        mc_per_cell=mc_per_cell,
    )
    ideal_dist = make_ideal_distribution(
        phit_true,
        target_prob=target_prob,
        variability=variability,
        total_trials=total_trials,
    )

    xlabels = ["shortest", "short", "medium", "long", "longest"]
    ylabels = ["closest", "close", "medium", "far", "farthest"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in [
        (axes[0], phit_true, "True P(hit) per cell"),
        (axes[1], ideal_dist, "Ideal distribution over (d,t) cells"),
    ]:
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(f"{title} - {profile_name}")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                txt = str(mat[i, j]) if mat.dtype == int else f"{mat[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig, axes, phit_true, ideal_dist


def _call_with_supported_kwargs(func, kwargs):
    sig = inspect.signature(func)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return func(**filtered)


def run_algorithm(algorithm_name, *args, patient_profile=None, **kwargs):
    module = _load_algorithm(algorithm_name)
    entrypoints = (
        "run_controller",
        "run_quest_plus_dt",
        "run_margin_controller",
        "run_staircase_sim",
        "run_sim",
    )
    for name in entrypoints:
        func = getattr(module, name, None)
        if callable(func):
            PatientModel = _load_patient_model()
            profile_params = PATIENT_PROFILES.get(patient_profile, {}) if patient_profile else {}
            call_kwargs = dict(kwargs)
            seed = int(call_kwargs.get("seed", profile_params.get("seed", 7)))
            rng = np.random.default_rng(seed)
            t_fixed = PATIENT_FIXED_T.get(patient_profile, T_RANDOM_MAX) if patient_profile else None

            class _FixedTimeRandomDirectionPatient(PatientModel):
                def __init__(self, *p_args, t_fixed, direction_rng, **p_kwargs):
                    super().__init__(*p_args, **p_kwargs)
                    self._t_fixed = float(t_fixed) if t_fixed is not None else None
                    self._dir_rng = direction_rng
                    self._executed_t = []

                def sample_trial(self, *, t_sys, d_sys, distance_level, previous_hit=True, direction_bin=None):
                    if self._t_fixed is not None:
                        t_sys = float(self._t_fixed)
                    self._executed_t.append(float(t_sys))
                    if direction_bin is None:
                        direction_bin = int(self._dir_rng.integers(0, 5))
                    return super().sample_trial(
                        t_sys=t_sys,
                        d_sys=d_sys,
                        distance_level=distance_level,
                        previous_hit=previous_hit,
                        direction_bin=direction_bin,
                    )

            if patient_profile is not None:
                call_kwargs["patient_profile"] = patient_profile
            if t_fixed is not None:
                call_kwargs["t_fixed"] = t_fixed
            patient = _FixedTimeRandomDirectionPatient(
                **profile_params,
                t_fixed=t_fixed,
                direction_rng=rng,
            )
            call_kwargs["patient"] = patient
            if args:
                result = func(*args, **call_kwargs)
            else:
                result = _call_with_supported_kwargs(func, call_kwargs)

            if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], dict):
                logs = result[0]
                executed_t = list(getattr(patient, "_executed_t", []))
                if executed_t and "t" in logs and len(logs.get("t", [])) == len(executed_t):
                    logs["t"] = executed_t
                if executed_t and "t_sys" in logs and len(logs.get("t_sys", [])) == len(executed_t):
                    logs["t_sys"] = executed_t

            return result

    raise AttributeError(
        f"No known run entrypoint found in {algorithm_name}. "
        f"Tried: {', '.join(entrypoints)}"
    )


def plot_caterpillar_means_by_algorithm_1var(
    profile_stats,
    title="Mean/SD by Patient Profile Across 1-var Algorithms",
    time_xlim=None,
    dist_xlim=None,
    save_path=None,
    show=False,
):
    if not isinstance(profile_stats, dict) or len(profile_stats) == 0:
        raise ValueError("profile_stats must be a non-empty dict")

    profile_names = list(profile_stats.keys())
    first_profile = profile_stats[profile_names[0]]
    algorithm_names = list(first_profile["algorithm_names"])
    n_algorithms = len(algorithm_names)
    n_profiles = len(profile_names)

    fig, axes = plt.subplots(
        n_algorithms,
        2,
        figsize=(10, max(3.5, 2.6 * n_algorithms)),
        sharex="col",
        squeeze=False,
    )

    for row_idx, algorithm_name in enumerate(algorithm_names):
        y = np.arange(n_profiles)
        means_time = [profile_stats[p]["means_time"][row_idx] for p in profile_names]
        std_time = [profile_stats[p]["std_time"][row_idx] for p in profile_names]
        means_dist = [profile_stats[p]["means_dist"][row_idx] for p in profile_names]
        std_dist = [profile_stats[p]["std_dist"][row_idx] for p in profile_names]

        panels = [
            ("Time", means_time, std_time, time_xlim),
            ("Distance", means_dist, std_dist, dist_xlim),
        ]

        for col_idx, (label, means, stds, xlim) in enumerate(panels):
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
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    assets_dir = BASE_DIR / "Assets" / "1_var"
    assets_dir.mkdir(parents=True, exist_ok=True)
    patient_profiles_dir = assets_dir / "patient_profiles"
    patient_profiles_dir.mkdir(parents=True, exist_ok=True)

    viz = _load_visualization_module()
    PatientModel = _load_patient_model()
    ideal_mod = _load_ideal_distribution_module()

    standard_d_min = float(ideal_mod["D_MIN"])
    standard_d_max = float(ideal_mod["D_MAX"])

    fig, _ = viz.plot_v_mean_by_profile(PATIENT_PROFILES, PatientModel)
    fig.savefig(patient_profiles_dir / "patient_v_mean_by_profile.png", dpi=150)
    plt.close(fig)

    fig, _ = viz.plot_d_by_profile(PATIENT_PROFILES, PatientModel)
    fig.savefig(patient_profiles_dir / "patient_rom_by_profile.png", dpi=150)
    plt.close(fig)

    algorithms = [
        "control_system_1var",
        "operations_research_1var",
        "staircasing_1var",
        "logistic_online_1var",
        "Qlearning_1var",
        "QUEST_1var",
    ]

    T_FIXED_VALUES = [0.8, 2, 4, 8]
    T_COLORS = {0.8: "#e41a1c", 2: "#ff7f00", 4: "#377eb8", 8: "#4daf4a"}

    profile_names = list(PATIENT_PROFILES.keys())
    n_profiles = len(profile_names)
    n_algorithms = len(algorithms)

    # Patient-specific phit/ideal plots (independent of t_fixed)
    for patient_profile in profile_names:
        fig, _, _, _ = plot_phit_and_ideal_by_profile(patient_profile)
        fig.savefig(assets_dir / f"{patient_profile}_phit_ideal.png", dpi=150)
        plt.close(fig)

    # ----------------------------------------------------------------
    # Run all experiments across t_fixed values
    # results_by_t[t_fixed][profile][algorithm] = {logs, counts, d_arr, t_arr, dir_counts}
    # ----------------------------------------------------------------
    results_by_t = {}
    _orig_patient_fixed_t = dict(PATIENT_FIXED_T)
    for t_fixed in T_FIXED_VALUES:
        for k in PATIENT_FIXED_T:
            PATIENT_FIXED_T[k] = float(t_fixed)
        results_by_t[t_fixed] = {}
        for patient_profile in profile_names:
            results_by_t[t_fixed][patient_profile] = {}
            print(f"=== t_fixed={t_fixed}s | {patient_profile} ===")
            for algorithm in algorithms:
                result = run_algorithm(
                    algorithm_name=algorithm,
                    patient_profile=patient_profile,
                    n_trials=200,
                    calibration=False,
                )
                logs, counts = result[0], result[1]
                d_arr = np.asarray(logs.get("d", []), dtype=float)
                t_arr = np.asarray(logs.get("t", []), dtype=float)
                dir_arr = np.asarray(logs.get("direction", []), dtype=int)
                dir_counts = np.zeros(5, dtype=int)
                for di in range(5):
                    if dir_arr.size > 0:
                        dir_counts[di] = int(np.sum(dir_arr == di))
                results_by_t[t_fixed][patient_profile][algorithm] = {
                    "logs": logs,
                    "counts": np.asarray(counts),
                    "d_arr": d_arr,
                    "t_arr": t_arr,
                    "dir_counts": dir_counts,
                }
                hr = np.mean(np.array(logs.get("hit", []))[-200:])
                avg_d, _ = viz.average_distance(logs)
                print(f"  {algorithm}: hit={hr:.3f}  mean_d={avg_d:.3f}")
    for k, v in _orig_patient_fixed_t.items():
        PATIENT_FIXED_T[k] = v

    # ----------------------------------------------------------------
    # Plot 1: Rolling hit rate matrix — one colored line per t_fixed
    # rows=profiles, cols=algorithms
    # ----------------------------------------------------------------
    window = 50
    fig, axes = plt.subplots(
        n_profiles, n_algorithms,
        figsize=(4 * n_algorithms, 2.8 * n_profiles),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, profile_name in enumerate(profile_names):
        for col, algorithm in enumerate(algorithms):
            ax = axes[row, col]
            for t_fixed in T_FIXED_VALUES:
                hits = results_by_t[t_fixed][profile_name][algorithm]["logs"].get("hit", [])
                rolling = viz.rolling_hitting_rate({"hit": hits}, window=window, min_periods=1)
                if rolling.size > 0:
                    ax.plot(rolling, linewidth=1.5, color=T_COLORS[t_fixed], label=f"t={t_fixed}s")
            for vx in (25, 50, 100):
                ax.axvline(x=vx, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(algorithm, fontsize=9)
            if col == 0:
                ax.set_ylabel(profile_name, fontsize=9)
            if row == n_profiles - 1:
                ax.set_xlabel("Trial")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9, ncol=len(T_FIXED_VALUES))
    fig.suptitle("Rolling Hit Rate (Profiles × Algorithms) — Multi t_fixed", fontsize=13)
    fig.tight_layout()
    fig.savefig(assets_dir / "all_algorithms_hit_rate_matrix_multi_t.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # Plot 2: d-level counts — grouped bars per (profile, algorithm)
    # ----------------------------------------------------------------
    level_labels = ["closest", "close", "medium", "far", "farthest"]
    n_levels = 5
    x = np.arange(n_levels)
    bar_width = 0.18
    bar_offsets = [(i - (len(T_FIXED_VALUES) - 1) / 2) * bar_width for i in range(len(T_FIXED_VALUES))]
    for profile_name in profile_names:
        for algorithm in algorithms:
            fig, ax = plt.subplots(figsize=(8, 3.8))
            for ti, t_fixed in enumerate(T_FIXED_VALUES):
                logs = results_by_t[t_fixed][profile_name][algorithm]["logs"]
                d_counts = _discretized_d_counts_from_logs(logs, standard_d_min, standard_d_max, n_levels)
                bars = ax.bar(
                    x + bar_offsets[ti], d_counts, bar_width,
                    label=f"t={t_fixed}s", color=T_COLORS[t_fixed],
                    edgecolor="black", linewidth=0.5,
                )
                for rect, val in zip(bars, d_counts):
                    if val > 0:
                        ax.text(
                            rect.get_x() + rect.get_width() * 0.5,
                            rect.get_height(), str(int(val)),
                            ha="center", va="bottom", fontsize=6,
                        )
            ax.set_xticks(x)
            ax.set_xticklabels(level_labels)
            ax.set_xlabel("Distance Level")
            ax.set_ylabel("Count")
            ax.set_title(f"d-level counts: {algorithm} — {profile_name}")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                assets_dir / f"{profile_name}_{algorithm}_d_levels_counts_multi_t.png",
                dpi=150,
            )
            plt.close(fig)

    # ----------------------------------------------------------------
    # Plot 3: Caterpillar matrix — rows=algorithms, cols=t_fixed
    # Each cell: patient profiles on y-axis, mean distance ± SD on x-axis
    # Color encodes t_fixed (matches the column)
    # ----------------------------------------------------------------
    n_t = len(T_FIXED_VALUES)
    y_prof = np.arange(n_profiles)

    fig, axes = plt.subplots(
        n_algorithms, n_t,
        figsize=(3.0 * n_t, 2.0 * n_algorithms),
        squeeze=False,
        sharex=True, sharey=True,
    )
    # Global x-range shared across all cells
    all_means_dist = [
        viz.average_distance(results_by_t[t][p][a]["logs"])[0]
        for t in T_FIXED_VALUES for p in profile_names for a in algorithms
    ]
    all_sd_dist = [
        viz.average_distance(results_by_t[t][p][a]["logs"])[1]
        for t in T_FIXED_VALUES for p in profile_names for a in algorithms
    ]
    x_lo = max(0.0, min(m - s for m, s in zip(all_means_dist, all_sd_dist)))
    x_hi = max(m + s for m, s in zip(all_means_dist, all_sd_dist)) * 1.05

    for row, algorithm in enumerate(algorithms):
        for col, t_fixed in enumerate(T_FIXED_VALUES):
            ax = axes[row, col]
            color = T_COLORS[t_fixed]
            for pi, profile_name in enumerate(profile_names):
                logs = results_by_t[t_fixed][profile_name][algorithm]["logs"]
                avg_d, sd_d = viz.average_distance(logs)
                ax.errorbar(
                    avg_d, y_prof[pi], xerr=sd_d,
                    fmt="o", capsize=3, linewidth=1.5, color=color,
                )
            ax.set_xlim(x_lo, x_hi)
            ax.set_yticks(y_prof)
            ax.grid(True, axis="x", alpha=0.3)
            if col == 0:
                ax.set_yticklabels(profile_names, fontsize=7)
                ax.set_ylabel(algorithm, fontsize=8)
            if row == 0:
                ax.set_title(f"t={t_fixed}s", fontsize=9, color=color)
            if row == n_algorithms - 1:
                ax.set_xlabel("Mean Distance (m)", fontsize=8)

    fig.suptitle("Mean Distance ± SD — Algorithms × t_fixed", fontsize=12)
    fig.tight_layout()
    fig.savefig(assets_dir / "all_algorithms_caterpillar_multi_t.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # Plot 4: Distance distribution matrix — rows=profiles, cols=algorithms
    # Each cell: grouped bar chart with one bar group per t_fixed (colored).
    # Since t is fixed per run, the old d×t heatmap t-axis carried no info;
    # this replaces the 4 separate figures with a single unified plot.
    # ----------------------------------------------------------------
    N_D_BINS = 5
    level_labels_short = ["cls", "close", "med", "far", "frst"]

    # Global d range across all t_fixed for consistent binning
    all_d_vals_global = [
        v
        for t_fixed in T_FIXED_VALUES
        for profile_name in profile_names
        for algorithm in algorithms
        for v in results_by_t[t_fixed][profile_name][algorithm]["d_arr"].tolist()
    ]
    global_d_lo = min(all_d_vals_global) if all_d_vals_global else 0.0
    global_d_hi = max(all_d_vals_global) if all_d_vals_global else 1.0
    if abs(global_d_hi - global_d_lo) < 1e-6:
        global_d_hi = global_d_lo + 0.1
    d_edges_global = np.linspace(global_d_lo, global_d_hi, N_D_BINS + 1)
    d_bin_centers = 0.5 * (d_edges_global[:-1] + d_edges_global[1:])
    d_bin_labels = [f"{v:.2f}" for v in d_bin_centers]

    def count_d_bins(d_arr, _de=d_edges_global):
        counts = np.zeros(N_D_BINS, dtype=int)
        for d_val in d_arr:
            bi = int(np.clip(np.searchsorted(_de, float(d_val), side="right") - 1, 0, N_D_BINS - 1))
            counts[bi] += 1
        return counts

    bar_w = 0.15
    t_offsets = [(i - (len(T_FIXED_VALUES) - 1) / 2) * bar_w for i in range(len(T_FIXED_VALUES))]
    x_bins = np.arange(N_D_BINS)

    fig_mat, axes_mat = plt.subplots(
        n_profiles, n_algorithms,
        figsize=(3.5 * n_algorithms, 2.8 * n_profiles),
        sharex=True, sharey=False,
        squeeze=False,
    )
    for row, profile_name in enumerate(profile_names):
        for col, algorithm in enumerate(algorithms):
            ax = axes_mat[row, col]
            for ti, t_fixed in enumerate(T_FIXED_VALUES):
                d_arr = results_by_t[t_fixed][profile_name][algorithm]["d_arr"]
                counts = count_d_bins(d_arr)
                ax.bar(
                    x_bins + t_offsets[ti], counts, bar_w,
                    color=T_COLORS[t_fixed], label=f"t={t_fixed}s",
                    edgecolor="black", linewidth=0.4, alpha=0.85,
                )
            ax.set_xticks(x_bins)
            ax.set_xticklabels(d_bin_labels, fontsize=6, rotation=45)
            ax.grid(True, axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(algorithm, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{profile_name}\nCount", fontsize=8)
            if row == n_profiles - 1:
                ax.set_xlabel("Distance bin (m)", fontsize=7)

    handles_mat, labels_mat = axes_mat[0, 0].get_legend_handles_labels()
    fig_mat.legend(handles_mat, labels_mat, loc="upper right", fontsize=9, ncol=len(T_FIXED_VALUES))
    fig_mat.suptitle(
        "Distance Distribution — Profiles × Algorithms (color = t_fixed)",
        fontsize=13, fontweight="bold",
    )
    fig_mat.tight_layout()
    fig_mat.savefig(assets_dir / "all_algorithms_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig_mat)
