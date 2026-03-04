import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

# 2-var simulation control panel for distance/time-only algorithms.
# Patient profiles are spatially uniform (no directional asymmetry).

PATIENT_PROFILES = {
    # ── Severity spectrum ────────────────────────────────────────────
    # Low ROM → strong rightie (high dir decay, can only reach right)
    # High ROM → neutral (low dir decay, can reach both sides)
    "overall_weak": {
        "k_d0_per_sec": 0.15,
        "k_d_decay": 1.0,
        "v_sigma0": 0.04,          # CV ~27%
        "v_sigma_growth": 0.04,
        "max_reach": 0.40,
        "handedness": 0.8,         # rightie — limited ROM = limited direction
        "k_dir_decay": 2.0,        # strong penalty reaching left
    },
    "overall_medium": {
        "k_d0_per_sec": 0.7,
        "k_d_decay": 0.55,
        "v_sigma0": 0.10,          # CV ~14%
        "v_sigma_growth": 0.024,
        "max_reach": 0.70,
        "handedness": 0.2,         # mild rightie — gentle directional bias
        "k_dir_decay": 0.5,        # ~10% speed cut at far-left (was 34%)
    },
    "overall_strong": {
        "k_d0_per_sec": 2.0,
        "k_d_decay": 0.1,
        "v_sigma0": 0.20,          # CV ~10%
        "v_sigma_growth": 0.008,
        "max_reach": 1.0,
        "handedness": 0.0,         # neutral — full ROM, reach both sides
        "k_dir_decay": 0.1,
    },

    # ── Speed / ROM dissociation ─────────────────────────────────────
    "highspeed_lowrom": {
        "k_d0_per_sec": 1.5,
        "k_d_decay": 0.1,
        "v_sigma0": 0.12,          # CV ~12%
        "v_sigma_growth": 0.01,
        "max_reach": 0.50,
        "handedness": 0.8,         # rightie — limited ROM = limited direction
        "k_dir_decay": 2.0,
    },
    "lowspeed_highrom": {
        "k_d0_per_sec": 0.2,
        "k_d_decay": 0.1,
        "v_sigma0": 0.06,          # CV ~20%
        "v_sigma_growth": 0.04,
        "max_reach": 1.0,
        "handedness": 0.0,         # neutral — full ROM, reach both sides
        "k_dir_decay": 0.1,
    },
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
    algorithms_dir = BASE_DIR / "2_var"
    path = algorithms_dir / f"{algorithm_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"No algorithm file found: {path}")
    module_name = f"two_var.{path.stem}"
    return _load_module_from_path(module_name, path)


def _load_visualization_module():
    module_path = BASE_DIR / "tests" / "visualization.py"
    return _load_module_from_path("tests.visualization", module_path)


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

            class _RandomizedDirectionPatient(PatientModel):
                def __init__(self, *p_args, dir_rng, **p_kwargs):
                    super().__init__(*p_args, **p_kwargs)
                    self._dir_rng = dir_rng
                    self._executed_direction = []

                def sample_trial(self, *, t_sys, d_sys, distance_level, previous_hit=True, direction_bin=None):
                    if direction_bin is None:
                        direction_bin = int(self._dir_rng.integers(0, 5))
                    direction_bin = int(np.clip(direction_bin, 0, 4))
                    self._executed_direction.append(direction_bin)
                    return super().sample_trial(
                        t_sys=t_sys,
                        d_sys=d_sys,
                        distance_level=distance_level,
                        previous_hit=previous_hit,
                        direction_bin=direction_bin,
                    )

            patient = _RandomizedDirectionPatient(**profile_params, dir_rng=rng)
            call_kwargs["patient"] = patient
            if args:
                result = func(*args, **call_kwargs)
            else:
                result = _call_with_supported_kwargs(func, call_kwargs)

            if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], dict):
                logs = result[0]
                executed_dir = list(getattr(patient, "_executed_direction", []))
                if executed_dir and "direction" in logs and len(logs.get("direction", [])) == len(executed_dir):
                    logs["direction"] = executed_dir

            return result

    raise AttributeError(
        f"No known run entrypoint found in {algorithm_name}. "
        f"Tried: {', '.join(entrypoints)}"
    )


def plot_caterpillar_means_by_algorithm_2var(
    profile_stats,
    title="Mean/SD by Patient Profile Across 2-var Algorithms",
    time_xlim=None,
    dist_xlim=None,
    dir_xlim=None,
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
        3,
        figsize=(14, max(3.5, 2.6 * n_algorithms)),
        sharex="col",
        squeeze=False,
    )

    for row_idx, algorithm_name in enumerate(algorithm_names):
        y = np.arange(n_profiles)
        means_time = [profile_stats[p]["means_time"][row_idx] for p in profile_names]
        std_time = [profile_stats[p]["std_time"][row_idx] for p in profile_names]
        means_dist = [profile_stats[p]["means_dist"][row_idx] for p in profile_names]
        std_dist = [profile_stats[p]["std_dist"][row_idx] for p in profile_names]
        means_dir = [profile_stats[p]["means_dir"][row_idx] for p in profile_names]
        std_dir = [profile_stats[p]["std_dir"][row_idx] for p in profile_names]

        panels = [
            ("Time", means_time, std_time, time_xlim),
            ("Distance", means_dist, std_dist, dist_xlim),
            ("Direction", means_dir, std_dir, dir_xlim),
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
    axes[-1, 2].set_xlabel("Direction (degrees)")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _five_bin_labels(values, fmt):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return ["n/a"] * 5
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, 6)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return [fmt.format(c) for c in centers]


if __name__ == "__main__":
    assets_dir = BASE_DIR / "Assets" / "2_var"
    assets_dir.mkdir(parents=True, exist_ok=True)
    patient_profiles_dir = assets_dir / "patient_profiles"
    patient_profiles_dir.mkdir(parents=True, exist_ok=True)

    viz = _load_visualization_module()
    PatientModel = _load_patient_model()

    fig, _ = viz.plot_v_mean_by_profile(PATIENT_PROFILES, PatientModel)
    fig.savefig(patient_profiles_dir / "patient_v_mean_by_profile.png", dpi=150)
    plt.close(fig)

    fig, _ = viz.plot_d_by_profile(PATIENT_PROFILES, PatientModel)
    fig.savefig(patient_profiles_dir / "patient_rom_by_profile.png", dpi=150)
    plt.close(fig)

    all_t_values = []
    all_d_values = []
    all_dir_values = []
    profile_caterpillar_stats = {}
    all_hits_by_profile = {}
    all_counts_by_profile = {}
    all_matrix_data = {}

    algorithms = [
        "control_system_2var",
        "operations_research_2var",
        "staircasing_2var",
        "logistic_online_2var",
        "Qlearning_2var",
        "QUEST_2var",
    ]

    for patient_profile in PATIENT_PROFILES.keys():
        print(f"=== Patient profile: {patient_profile} ===")

        fig, _, _, _ = plot_phit_and_ideal_by_profile(patient_profile)
        fig.savefig(assets_dir / f"{patient_profile}_phit_ideal.png", dpi=150)
        plt.close(fig)

        means_time = []
        means_dist = []
        means_dir = []
        std_time = []
        std_dist = []
        std_dir = []
        hits_by_algorithm = {}
        counts_by_algorithm = {}

        for algorithm in algorithms:
            print(f"Running algorithm: {algorithm} (patient={patient_profile})")
            result = run_algorithm(
                algorithm_name=algorithm,
                patient_profile=patient_profile,
                n_trials=200,
                calibration=True,
            )
            if not isinstance(result, tuple) or len(result) < 2:
                raise ValueError(f"Unexpected result from {algorithm}: {type(result)}")
            logs, counts = result[0], result[1]

            avg_time, sd_time = viz.average_time(logs)
            avg_dist, sd_dist = viz.average_distance(logs)
            avg_dir, sd_dir_val = viz.average_direction(logs)

            means_time.append(avg_time)
            means_dist.append(avg_dist)
            means_dir.append(avg_dir)
            std_time.append(sd_time)
            std_dist.append(sd_dist)
            std_dir.append(sd_dir_val)
            hits_by_algorithm[algorithm] = logs.get("hit", [])
            counts_by_algorithm[algorithm] = np.asarray(counts)

            print(f"  Mean hit rate (last 200): {np.mean(np.array(logs.get('hit', []))[-200:]):.3f}")
            print(f"  Mean time: {avg_time:.4f} +/- {sd_time:.4f}")
            print(f"  Mean dist: {avg_dist:.4f} +/- {sd_dist:.4f}")
            print(f"  Mean dir:  {avg_dir:.4f} +/- {sd_dir_val:.4f}")

            t_arr = np.asarray(logs.get("t", []), dtype=float)
            d_arr = np.asarray(logs.get("d", []), dtype=float)
            dir_arr = np.asarray(logs.get("direction", []), dtype=float)
            if t_arr.size > 0:
                all_t_values.extend(t_arr.tolist())
            if d_arr.size > 0:
                all_d_values.extend(d_arr.tolist())
            if dir_arr.size > 0:
                all_dir_values.extend([
                    float(viz.DIR_INDEX_TO_ANGLE.get(int(d), float(d))) for d in dir_arr.tolist()
                ])

            dirs = np.array(logs.get("direction", []), dtype=int)
            dir_counts = np.zeros(5, dtype=int)
            for di in range(5):
                dir_counts[di] = int(np.sum(dirs == di))

            if patient_profile not in all_matrix_data:
                all_matrix_data[patient_profile] = {}
            all_matrix_data[patient_profile][algorithm] = {
                "counts": np.asarray(counts),
                "dir_counts": dir_counts,
                "d_arr": d_arr,
                "t_arr": t_arr,
                "d_range": (float(d_arr.min()), float(d_arr.max())) if len(d_arr) > 0 else (0.1, 0.8),
                "t_range": (float(t_arr.min()), float(t_arr.max())) if len(t_arr) > 0 else (1.0, 7.0),
            }

        profile_caterpillar_stats[patient_profile] = {
            "algorithm_names": list(algorithms),
            "means_time": list(means_time),
            "std_time": list(std_time),
            "means_dist": list(means_dist),
            "std_dist": list(std_dist),
            "means_dir": list(means_dir),
            "std_dir": list(std_dir),
        }
        all_hits_by_profile[patient_profile] = dict(hits_by_algorithm)
        all_counts_by_profile[patient_profile] = dict(counts_by_algorithm)

        viz.plot_rolling_hit_rate(
            hits_by_algorithm,
            window=50,
            min_periods=1,
            title=f"Rolling Hit Rate by Algorithm - {patient_profile}",
            save_path=assets_dir / f"{patient_profile}_rolling_hit_rate.png",
            show=False,
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

    plot_caterpillar_means_by_algorithm_2var(
        profile_stats=profile_caterpillar_stats,
        title="Mean/SD by Patient Profile Across 2-var Algorithms",
        time_xlim=time_xlim,
        dist_xlim=dist_xlim,
        dir_xlim=dir_xlim,
        save_path=assets_dir / "all_algorithms_caterpillar.png",
        show=False,
    )

    viz.plot_hit_rate_matrix(
        all_hits_by_profile=all_hits_by_profile,
        algorithms=algorithms,
        window=50,
        save_path=assets_dir / "all_algorithms_hit_rate_matrix.png",
        show=False,
    )

    viz.plot_counts_matrix(
        all_counts_by_profile=all_counts_by_profile,
        algorithms=algorithms,
        xlabels=["shortest", "short", "medium", "long", "longest"],
        ylabels=["closest", "close", "medium", "far", "farthest"],
        title="Counts Matrix (Rows=Profiles, Cols=Algorithms)",
        save_path=assets_dir / "all_algorithms_counts_matrix.png",
        show=False,
        annotate=True,
    )

    # Combined matrices: d×t heatmaps + direction bars.
    dir_angle_labels = [f"{viz.DIR_INDEX_TO_ANGLE[d]:.0f}" for d in range(5)]
    profile_names = list(all_hits_by_profile.keys())
    n_profiles = len(profile_names)
    n_algorithms = len(algorithms)

    global_d_lo = min(mdata["d_range"][0] for prof in all_matrix_data.values() for mdata in prof.values())
    global_d_hi = max(mdata["d_range"][1] for prof in all_matrix_data.values() for mdata in prof.values())
    global_t_lo = min(mdata["t_range"][0] for prof in all_matrix_data.values() for mdata in prof.values())
    global_t_hi = max(mdata["t_range"][1] for prof in all_matrix_data.values() for mdata in prof.values())

    N_D_BINS = 5
    N_T_BINS = 8
    d_edges = np.linspace(global_d_lo, global_d_hi, N_D_BINS + 1)
    t_edges = np.geomspace(max(global_t_lo, 0.01), global_t_hi, N_T_BINS + 1)

    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])
    t_centers = np.sqrt(t_edges[:-1] * t_edges[1:])
    d_labels = [f"{v:.2f}" for v in d_centers]
    t_labels = [f"{v:.1f}" for v in t_centers]

    def rebin_dt(d_arr, t_arr):
        mat = np.zeros((N_D_BINS, N_T_BINS), dtype=int)
        for d_val, t_val in zip(d_arr, t_arr):
            di = int(np.clip(np.searchsorted(d_edges, float(d_val), side='right') - 1, 0, N_D_BINS - 1))
            ti = int(np.clip(np.searchsorted(t_edges, float(t_val), side='right') - 1, 0, N_T_BINS - 1))
            mat[di, ti] += 1
        return mat

    fig_mat = plt.figure(figsize=(6.5 * n_algorithms, 4.5 * n_profiles))
    outer_gs = gridspec.GridSpec(n_profiles, n_algorithms, figure=fig_mat, wspace=0.35, hspace=0.5)

    for row, profile_name in enumerate(profile_names):
        for col, algorithm in enumerate(algorithms):
            mdata = all_matrix_data.get(profile_name, {}).get(algorithm)
            if mdata is None:
                continue

            counts_mat = rebin_dt(mdata["d_arr"], mdata["t_arr"])
            dir_counts = mdata["dir_counts"]

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[row, col], height_ratios=[5, 1.2], hspace=0.2
            )

            ax_heat = fig_mat.add_subplot(inner_gs[0])
            ax_heat.imshow(counts_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=max(counts_mat.max(), 1))
            ax_heat.set_xticks(range(N_T_BINS))
            ax_heat.set_xticklabels(t_labels, fontsize=6, rotation=45)
            ax_heat.set_yticks(range(N_D_BINS))
            ax_heat.set_yticklabels(d_labels, fontsize=7)
            for i in range(N_D_BINS):
                for j in range(N_T_BINS):
                    val = counts_mat[i, j]
                    if val > 0:
                        ax_heat.text(j, i, str(val), ha='center', va='center', fontsize=7, color='black')

            if row == 0:
                ax_heat.set_title(f"{algorithm}", fontsize=9, fontweight='bold')
            if col == 0:
                ax_heat.set_ylabel(f"{profile_name}\nDistance (m)", fontsize=8)
            else:
                ax_heat.set_ylabel("Distance (m)", fontsize=7)
            ax_heat.set_xlabel("Time (s)", fontsize=7)

            ax_dir = fig_mat.add_subplot(inner_gs[1])
            dir_mat = dir_counts.reshape(1, 5)
            ax_dir.imshow(dir_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=max(dir_counts.max(), 1))
            ax_dir.set_xticks(range(5))
            ax_dir.set_xticklabels(dir_angle_labels, fontsize=7)
            ax_dir.set_yticks([])
            for j in range(5):
                ax_dir.text(j, 0, str(dir_counts[j]), ha='center', va='center', fontsize=8, color='black')
            if col == 0:
                ax_dir.set_ylabel("Dir", fontsize=8)
            ax_dir.set_xlabel("Direction (deg)", fontsize=7)

    fig_mat.suptitle(
        "Target Distribution: Distance×Time & Direction — All Algorithms × Profiles",
        fontsize=14,
        fontweight='bold',
    )
    fig_mat.savefig(assets_dir / "all_algorithms_matrices.png", dpi=150)
    plt.close(fig_mat)
