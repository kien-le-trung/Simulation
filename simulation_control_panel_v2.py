import ast
import importlib.util
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# A simulation control panel for managing simulation parameters and execution.
PATIENT_PROFILES = {
    "overall_weak": {
        "k_d0_per_sec": 0.20,
        "k_d_decay": 4.0,
        "v_sigma0": 0.06,
        "v_sigma_growth": 0.04,
        # 3x3 map flattened by idx = elevation * 3 + azimuth
        # elevation low->high, azimuth left->right
        "spatial_strength_map": [1.0] * 9,
    },
    "overall_medium": {
        "k_d0_per_sec": 0.30,
        "k_d_decay": 1.0,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.03,
        "spatial_strength_map": [1.0] * 9,
    },

    "overall_strong": {
        "k_d0_per_sec": 0.70,
        "k_d_decay": 0.1,
        "v_sigma0": 0.03,
        "v_sigma_growth": 0.01,
        "spatial_strength_map": [1.0] * 9,
    },

    "highspeed_lowrom": {
        "k_d0_per_sec": 0.70,
        "k_d_decay": 4.0,
        "v_sigma0": 0.03,
        "v_sigma_growth": 0.01,
        "spatial_strength_map": [1.0] * 9,
    },

    "lowspeed_highrom": {
        "k_d0_per_sec": 0.20,
        "k_d_decay": 0.1,
        "v_sigma0": 0.06,
        "v_sigma_growth": 0.04,
        "spatial_strength_map": [1.0] * 9,
    },
}

def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
    algorithms_dir = BASE_DIR / "algorithms"
    path = algorithms_dir / f"{algorithm_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"No algorithm file found: {path}")
    module_name = f"algorithms.{path.stem}"
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
    assets_dir = BASE_DIR / "Assets"
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

    default_patient = PatientModel()
    time_xlim = (
        float(0),
        float(np.max(default_patient.t_levels)),
    )
    dist_xlim = (
        float(0),
        float(np.max(default_patient.d_levels)),
    )
    dir_az_angles = np.array(list(viz.DIR_INDEX_TO_AZIMUTH_ANGLE.values()), dtype=float)
    dir_el_angles = np.array(list(viz.DIR_INDEX_TO_ELEVATION_ANGLE.values()), dtype=float)
    dir_az_xlim = (float(np.min(dir_az_angles)), float(np.max(dir_az_angles)))
    dir_el_xlim = (float(np.min(dir_el_angles)), float(np.max(dir_el_angles)))
    profile_caterpillar_stats = {}

    algorithms = [
        "control_system_3var",
        "operations_research_3var",
        "staircasing_3var",
        "logistic_online_3var_v2",
    ]

    for patient_profile in PATIENT_PROFILES.keys():
        print(f"=== Patient profile: {patient_profile} ===")

        fig, axes, phit_true, ideal_dist = plot_phit_and_ideal_by_profile(patient_profile)
        fig.savefig(assets_dir / f"{patient_profile}_phit_ideal.png", dpi=150)
        plt.close(fig)

        means_time = []
        means_dist = []
        means_dir_az = []
        means_dir_el = []
        std_time = []
        std_dist = []
        std_dir_az = []
        std_dir_el = []
        hits_by_algorithm = {}

        for algorithm in algorithms:
            print(f"Running algorithm: {algorithm} (patient={patient_profile})")
            result = run_algorithm(
                algorithm_name=algorithm,
                patient_profile=patient_profile,
                n_trials=200,
                calibration=False,
            )
            if not isinstance(result, tuple) or len(result) < 2:
                raise ValueError(f"Unexpected result from {algorithm}: {type(result)}")
            logs, counts = result[0], result[1]

            avg_time, sd_time = viz.average_time(logs)
            avg_dist, sd_dist = viz.average_distance(logs)
            avg_dir_az, sd_dir_az, avg_dir_el, sd_dir_el = viz.average_direction_components(logs)

            means_time.append(avg_time)
            means_dist.append(avg_dist)
            means_dir_az.append(avg_dir_az)
            means_dir_el.append(avg_dir_el)
            std_time.append(sd_time)
            std_dist.append(sd_dist)
            std_dir_az.append(sd_dir_az)
            std_dir_el.append(sd_dir_el)
            hits_by_algorithm[algorithm] = logs.get("hit", [])
            print(f"  Mean hit rate (last 200): {np.mean(np.array(logs.get('hit', []))[-200:]):.3f}")
            print(f"  Mean time: {avg_time:.4f} +/- {sd_time:.4f}")
            print(f"  Mean dist: {avg_dist:.4f} +/- {sd_dist:.4f}")
            print(f"  Mean az:   {avg_dir_az:.4f} +/- {sd_dir_az:.4f}")
            print(f"  Mean el:   {avg_dir_el:.4f} +/- {sd_dir_el:.4f}")

            viz.plot_heatmap(
                counts,
                title=f"Counts heatmap for {algorithm} - {patient_profile}",
                xlabels=["shortest", "short", "medium", "long", "longest"],
                ylabels=["closest", "close", "medium", "far", "farthest"],
                save_path=assets_dir / f"{patient_profile}_{algorithm}_counts_heatmap.png",
                show=False,
            )

        profile_caterpillar_stats[patient_profile] = {
            "algorithm_names": list(algorithms),
            "means_time": list(means_time),
            "std_time": list(std_time),
            "means_dist": list(means_dist),
            "std_dist": list(std_dist),
            "means_dir_az": list(means_dir_az),
            "std_dir_az": list(std_dir_az),
            "means_dir_el": list(means_dir_el),
            "std_dir_el": list(std_dir_el),
        }
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
        dir_az_xlim=dir_az_xlim,
        dir_el_xlim=dir_el_xlim,
        save_path=assets_dir / "all_algorithms_caterpillar.png",
        show=False,
    )
    
