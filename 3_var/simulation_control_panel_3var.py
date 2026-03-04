import ast
import importlib.util
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

# A simulation control panel for managing simulation parameters and execution.
#
# Design philosophy:
#   Speed (k_d0_per_sec, k_d_decay) and ROM (max_reach) are DECOUPLED.
#   - k_d_decay models mild speed reduction with distance (Fitts'-law-like).
#   - max_reach is the hard geometric/structural ROM ceiling (scalar, meters).
#   - handedness (-1 leftie to +1 rightie) creates a directional gradient:
#     the patient is faster/more accurate on the preferred side.

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

    # # ── Handedness profiles ──────────────────────────────────────────
    # "rightie": {
    #     "k_d0_per_sec": 0.30,
    #     "k_d_decay": 0.5,
    #     "v_sigma0": 0.04,
    #     "v_sigma_growth": 0.025,
    #     "max_reach": 0.70,
    #     "handedness": 0.8,
    #     "k_dir_decay": 1.0,
    #     "k_dir_noise": 0.5,
    # },
    # "leftie": {
    #     "k_d0_per_sec": 0.30,
    #     "k_d_decay": 0.5,
    #     "v_sigma0": 0.04,
    #     "v_sigma_growth": 0.025,
    #     "max_reach": 0.70,
    #     "handedness": -0.8,
    #     "k_dir_decay": 1.0,
    #     "k_dir_noise": 0.5,
    # },
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
    algorithms_dir = BASE_DIR / "3_var"
    path = algorithms_dir / f"{algorithm_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"No algorithm file found: {path}")
    module_name = f"three_var.{path.stem}"
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
                if mat.dtype == int:
                    txt = str(mat[i, j])
                else:
                    txt = f"{mat[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig, axes, phit_true, ideal_dist


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
            kwargs = dict(kwargs)
            kwargs["patient"] = PatientModel(**profile_params)
            return func(*args, **kwargs)

    raise AttributeError(
        f"No known run entrypoint found in {algorithm_name}. "
        f"Tried: {', '.join(entrypoints)}"
    )

# plot_v_mean_by_profile()
# plt.show()

if __name__ == "__main__":
    assets_dir = BASE_DIR / "Assets" / "3_var"
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

    # fig, _ = viz.plot_direction_by_profile(PATIENT_PROFILES, PatientModel)
    # fig.savefig(patient_profiles_dir / "patient_direction_profile.png", dpi=150)
    # plt.close(fig)

    default_patient = PatientModel()
    time_xlim = (
        float(0),
        float(np.max(default_patient.t_levels)),
    )
    dist_xlim = (
        float(0),
        float(np.max(default_patient.d_levels)),
    )
    dir_angles = np.array(list(viz.DIR_INDEX_TO_ANGLE.values()), dtype=float)
    dir_xlim = (float(np.min(dir_angles)), float(np.max(dir_angles)))
    profile_caterpillar_stats = {}
    all_hits_by_profile = {}
    all_matrix_data = {}

    algorithms = [
        "control_system_3var",
        "operations_research_3var",
        "staircasing_3var",
        "logistic_online_3var_v2",
        # "QUEST_3var",
        # "Qlearning",
        # "hybrid_adaptive_3var",
    ]

    for patient_profile in PATIENT_PROFILES.keys():
        print(f"=== Patient profile: {patient_profile} ===")

        fig, axes, phit_true, ideal_dist = plot_phit_and_ideal_by_profile(patient_profile)
        fig.savefig(assets_dir / f"{patient_profile}_phit_ideal.png", dpi=150)
        plt.close(fig)

        means_time = []
        means_dist = []
        means_dir = []
        std_time = []
        std_dist = []
        std_dir = []
        hits_by_algorithm = {}

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
            print(f"  Mean hit rate (last 200): {np.mean(np.array(logs.get('hit', []))[-200:]):.3f}")
            print(f"  Mean time: {avg_time:.4f} +/- {sd_time:.4f}")
            print(f"  Mean dist: {avg_dist:.4f} +/- {sd_dist:.4f}")
            print(f"  Mean dir:  {avg_dir:.4f} +/- {sd_dir_val:.4f}")

            # Store data for combined matrices plot
            dirs = np.array(logs.get("direction", []))
            dir_counts = np.zeros(5, dtype=int)
            for di in range(5):
                dir_counts[di] = int(np.sum(dirs == di))
            d_arr = np.array(logs.get("d", []))
            t_arr = np.array(logs.get("t", []))
            if patient_profile not in all_matrix_data:
                all_matrix_data[patient_profile] = {}
            all_matrix_data[patient_profile][algorithm] = {
                "counts": counts,
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
        viz.plot_rolling_hit_rate(
            hits_by_algorithm,
            window=50,
            min_periods=1,
            title=f"Rolling Hit Rate by Algorithm - {patient_profile}",
            save_path=assets_dir / f"{patient_profile}_rolling_hit_rate.png",
            show=False,
        )

    viz.plot_caterpillar_means_by_algorithm(
        profile_stats=profile_caterpillar_stats,
        title="Mean/SD by Patient Profile Across Algorithms",
        time_xlim=time_xlim,
        dist_xlim=dist_xlim,
        dir_xlim=dir_xlim,
        save_path=assets_dir / "all_algorithms_caterpillar.png",
        show=False,
    )

    # ── Combined rolling hit rate grid: rows=profiles, cols=algorithms ───
    profile_names = list(all_hits_by_profile.keys())
    n_profiles = len(profile_names)
    n_algorithms = len(algorithms)
    fig, axes = plt.subplots(
        n_profiles,
        n_algorithms,
        figsize=(5 * n_algorithms, 3.5 * n_profiles),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    TARGET_HIT_RATE = 0.70
    CHECKPOINTS = [50, 100, 200]
    window = 50
    
    # Track errors per checkpoint for column-level means
    # col_errors[algorithm][checkpoint] = list of errors across profiles
    col_errors = {a: {cp: [] for cp in CHECKPOINTS} for a in algorithms}
    for row, profile_name in enumerate(profile_names):
        for col, algorithm in enumerate(algorithms):
            ax = axes[row, col]
            hits = all_hits_by_profile[profile_name].get(algorithm, [])
            hits_arr = np.array(hits)
            rolling = viz.rolling_hitting_rate({"hit": hits}, window=window, min_periods=1)
            ax.plot(rolling, linewidth=1.5)
            ax.axhline(y=TARGET_HIT_RATE, color='black', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            # Annotate MAE at each checkpoint, aligned along x-axis
            if len(hits_arr) > 0:
                for cp_i, cp in enumerate(CHECKPOINTS):
                    if len(hits_arr) >= cp:
                        hr = float(np.mean(hits_arr[:cp]))
                        err = abs(hr - TARGET_HIT_RATE)
                        col_errors[algorithm][cp].append(err)
                        ax.axvline(x=cp, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
                        ax.text(cp, 0.03, f"{err:.2f}",
                                transform=ax.get_xaxis_transform(),
                                fontsize=9, ha='center', va='bottom',
                                color='red', fontweight='bold')
            if row == 0:
                ax.set_title(algorithm, fontsize=9)
            if col == 0:
                ax.set_ylabel(profile_name, fontsize=9)
            if row == n_profiles - 1:
                ax.set_xlabel("Trial")
    # Add mean MAE per algorithm at all checkpoints in column header
    for col, algorithm in enumerate(algorithms):
        title_parts = [algorithm]
        mae_parts = []
        for cp in CHECKPOINTS:
            errs = col_errors[algorithm].get(cp, [])
            if errs:
                mae_parts.append(f"@{cp}={np.mean(errs):.3f}")
        if mae_parts:
            title_parts.append("MAE " + "  ".join(mae_parts))
        axes[0, col].set_title("\n".join(title_parts), fontsize=8)
    fig.suptitle("Rolling Hit Rate — All Profiles × All Algorithms", fontsize=13)
    fig.tight_layout()
    fig.savefig(assets_dir / "all_algorithms_hit_rate.png", dpi=150)
    plt.close(fig)

    # ── Combined matrices: d×t heatmaps + direction bars ────────────────
    dir_angle_labels = [f"{viz.DIR_INDEX_TO_ANGLE[d]:.0f}" for d in range(5)]

    # Compute global d/t ranges across all algorithms and profiles for consistent axes
    global_d_lo = min(mdata["d_range"][0] for prof in all_matrix_data.values() for mdata in prof.values())
    global_d_hi = max(mdata["d_range"][1] for prof in all_matrix_data.values() for mdata in prof.values())
    global_t_lo = min(mdata["t_range"][0] for prof in all_matrix_data.values() for mdata in prof.values())
    global_t_hi = max(mdata["t_range"][1] for prof in all_matrix_data.values() for mdata in prof.values())

    # Log-spaced time bins (more resolution at short times where data is dense)
    N_D_BINS = 5
    N_T_BINS = 8
    d_edges = np.linspace(global_d_lo, global_d_hi, N_D_BINS + 1)
    t_edges = np.geomspace(max(global_t_lo, 0.01), global_t_hi, N_T_BINS + 1)

    # Bin center labels
    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])
    t_centers = np.sqrt(t_edges[:-1] * t_edges[1:])  # geometric mean for log bins
    d_labels = [f"{v:.2f}" for v in d_centers]
    t_labels = [f"{v:.1f}" for v in t_centers]

    def rebin_dt(d_arr, t_arr):
        """Re-bin raw d/t arrays into N_D_BINS x N_T_BINS using log-spaced time edges."""
        mat = np.zeros((N_D_BINS, N_T_BINS), dtype=int)
        for d_val, t_val in zip(d_arr, t_arr):
            di = int(np.clip(np.searchsorted(d_edges, float(d_val), side='right') - 1, 0, N_D_BINS - 1))
            ti = int(np.clip(np.searchsorted(t_edges, float(t_val), side='right') - 1, 0, N_T_BINS - 1))
            mat[di, ti] += 1
        return mat

    fig_mat = plt.figure(figsize=(6.5 * n_algorithms, 4.5 * n_profiles))
    outer_gs = gridspec.GridSpec(n_profiles, n_algorithms, figure=fig_mat,
                                 wspace=0.35, hspace=0.5)

    for row, profile_name in enumerate(profile_names):
        for col, algorithm in enumerate(algorithms):
            mdata = all_matrix_data.get(profile_name, {}).get(algorithm)
            if mdata is None:
                continue

            # Re-bin raw data with log-spaced time bins
            counts_mat = rebin_dt(mdata["d_arr"], mdata["t_arr"])
            dir_counts = mdata["dir_counts"]

            # Inner grid: heatmap (top) + direction bar (bottom)
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[row, col],
                height_ratios=[5, 1.2], hspace=0.2)

            # Distance × Time heatmap (log-spaced time axis)
            ax_heat = fig_mat.add_subplot(inner_gs[0])
            im = ax_heat.imshow(counts_mat, aspect='auto', cmap='YlOrRd',
                                vmin=0, vmax=max(counts_mat.max(), 1))
            ax_heat.set_xticks(range(N_T_BINS))
            ax_heat.set_xticklabels(t_labels, fontsize=6, rotation=45)
            ax_heat.set_yticks(range(N_D_BINS))
            ax_heat.set_yticklabels(d_labels, fontsize=7)
            for i in range(N_D_BINS):
                for j in range(N_T_BINS):
                    val = counts_mat[i, j]
                    if val > 0:
                        ax_heat.text(j, i, str(val), ha='center', va='center',
                                     fontsize=7, color='black')

            if row == 0:
                ax_heat.set_title(f"{algorithm}", fontsize=9, fontweight='bold')
            if col == 0:
                ax_heat.set_ylabel(f"{profile_name}\nDistance (m)", fontsize=8)
            else:
                ax_heat.set_ylabel("Distance (m)", fontsize=7)
            ax_heat.set_xlabel("Time (s)", fontsize=7)

            # 1×5 direction bar (horizontal)
            ax_dir = fig_mat.add_subplot(inner_gs[1])
            dir_mat = dir_counts.reshape(1, 5)
            ax_dir.imshow(dir_mat, aspect='auto', cmap='YlOrRd',
                          vmin=0, vmax=max(dir_counts.max(), 1))
            ax_dir.set_xticks(range(5))
            ax_dir.set_xticklabels(dir_angle_labels, fontsize=7)
            ax_dir.set_yticks([])
            for j in range(5):
                ax_dir.text(j, 0, str(dir_counts[j]), ha='center', va='center',
                            fontsize=8, color='black')
            if col == 0:
                ax_dir.set_ylabel("Dir", fontsize=8)
            ax_dir.set_xlabel("Direction (deg)", fontsize=7)

    fig_mat.suptitle("Target Distribution: Distance×Time & Direction — All Algorithms × Profiles",
                     fontsize=14, fontweight='bold')
    fig_mat.savefig(assets_dir / "all_algorithms_matrices.png", dpi=150)
    plt.close(fig_mat)

    # ── Computation Time Benchmark ──────────────────────────────────────
    bench_profile = "overall_medium"
    bench_trials = [50, 100, 200]
    timing_data = {alg: [] for alg in algorithms}

    print("\n=== Computation Time Benchmark ===")
    for n_t in bench_trials:
        for alg in algorithms:
            t0 = time.perf_counter()
            run_algorithm(algorithm_name=alg, patient_profile=bench_profile,
                          n_trials=n_t, calibration=True)
            elapsed = time.perf_counter() - t0
            timing_data[alg].append(elapsed)
            print(f"  {alg:30s}  {n_t:4d} trials  {elapsed:7.3f}s")

    # Grouped bar chart
    fig_time, ax_time = plt.subplots(figsize=(max(8, 1.5 * len(algorithms)), 5))
    x = np.arange(len(algorithms))
    bar_width = 0.25
    for idx, n_t in enumerate(bench_trials):
        times = [timing_data[alg][idx] for alg in algorithms]
        bars = ax_time.bar(x + idx * bar_width, times, bar_width,
                           label=f"{n_t} trials")
        for bar, val in zip(bars, times):
            ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{val:.2f}s", ha='center', va='bottom', fontsize=7)

    ax_time.set_xticks(x + bar_width)
    ax_time.set_xticklabels([a.replace("_", "\n") for a in algorithms],
                            fontsize=8)
    ax_time.set_ylabel("Wall-clock time (seconds)")
    ax_time.set_title(f"Algorithm Computation Time ({bench_profile} profile)")
    ax_time.legend()
    ax_time.set_ylim(bottom=0)
    fig_time.tight_layout()
    fig_time.savefig(assets_dir / "algorithm_computation_time.png", dpi=150)
    plt.close(fig_time)

    # ── Hit Rate Error Summary Tables (per checkpoint) ─────────────────
    short_names = [a[:12] for a in algorithms]
    for cp in CHECKPOINTS:
        print(f"\n{'=' * 72}")
        print(f"  Hit Rate MAE vs Target ({TARGET_HIT_RATE:.2f})  |  first {cp} trials")
        print(f"{'=' * 72}")

        header = f"{'Profile':<22}" + "".join(f"{n:>13}" for n in short_names)
        print(header)
        print("-" * 72)

        cp_means = {a: [] for a in algorithms}
        for profile_name in profile_names:
            row_str = f"{profile_name:<22}"
            for algorithm in algorithms:
                hits = all_hits_by_profile[profile_name].get(algorithm, [])
                hits_arr = np.array(hits)
                if len(hits_arr) >= cp:
                    hr = float(np.mean(hits_arr[:cp]))
                    err = abs(hr - TARGET_HIT_RATE)
                    cp_means[algorithm].append(err)
                    row_str += f"{err:>13.3f}"
                else:
                    row_str += f"{'N/A':>13}"
            print(row_str)

        print("-" * 72)
        mean_row = f"{'MEAN':<22}"
        for algorithm in algorithms:
            errs = cp_means[algorithm]
            if errs:
                mean_row += f"{np.mean(errs):>13.3f}"
            else:
                mean_row += f"{'N/A':>13}"
        print(mean_row)
        print(f"{'=' * 72}")
