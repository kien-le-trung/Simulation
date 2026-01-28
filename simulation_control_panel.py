import ast
import importlib.util
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# A simulation control panel for managing simulation parameters and execution.
PATIENT_PROFILES = {
    "overall_weak" : {
        "k_d0_per_sec": 0.10,
        "k_d_decay": 0.1,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.03
    },
    "overall_medium" : {
        "k_d0_per_sec": 0.15,
        "k_d_decay": 0.1,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.03
    },
    "overall_strong" : {
        "k_d0_per_sec": 0.25,
        "k_d_decay": 0.1,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.03
    },
    "highspeed_lowrom" : {
        "k_d0_per_sec": 0.45,
        "k_d_decay": 0.5,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.05
    },
    "lowspeed_highrom" : {
        "k_d0_per_sec": 0.10,
        "k_d_decay": 0.01,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.02
    }
}

ALGORITHMS = ["operations_research",
              "QUEST_v2",
              "staircasing",
              "RL_logistic_index_simplified",
              "control_system_v2"]

def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_patient_model():
    module_path = BASE_DIR / "patients" / "patient_simulation_v3.py"
    module = _load_module_from_path("patients.patient_simulation_v3", module_path)
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


def plot_v_mean_by_profile(t_min=None, t_max=None, t_steps=200):
    PatientModel = _load_patient_model()
    if t_min is None or t_max is None:
        base = PatientModel()
        if t_min is None:
            t_min = float(np.min(base.t_levels))
        if t_max is None:
            t_max = float(np.max(base.t_levels))

    t_grid = np.linspace(float(t_min), float(t_max), int(t_steps))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, params in PATIENT_PROFILES.items():
        model = PatientModel(**params)
        v_mean = model._mean_speed(t_grid)
        ax.plot(t_grid, v_mean, label=name, linewidth=2.0)

    ax.set_title("v_mean vs t_sys by patient profile")
    ax.set_xlabel("t_sys")
    ax.set_ylabel("v_mean (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_d_by_profile(t_levels=None):
    PatientModel = _load_patient_model()
    if t_levels is None:
        base = PatientModel()
        t_levels = base.t_levels
    else:
        t_levels = np.array(t_levels, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, params in PATIENT_PROFILES.items():
        model = PatientModel(t_levels=t_levels, **params)
        ax.plot(model.t_levels, model.d_means, label=name, linewidth=2.0)

    ax.set_title("d_means vs t_levels by patient profile")
    ax.set_xlabel("t_levels")
    ax.set_ylabel("d_means (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax


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
        patient_speed_sd=patient_params.get("v_sigma0", 0.04),
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
    return fig, axes


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

plot_d_by_profile()
plt.show()
plot_v_mean_by_profile()
plt.show()

if __name__ == "__main__":
    # for profile in PATIENT_PROFILES.keys():
    #     plot_phit_and_ideal_by_profile(profile)
    #     plt.show()
    
    for algorithm in ALGORITHMS:
        print(f"Running algorithm: {algorithm}")   
        logs, counts = run_algorithm(algorithm_name=algorithm,
                                    patient_profile="lowspeed_highrom")
        plot_heatmap = _load_visualization_module().plot_heatmap
        # plot_heatmap(counts,
        #              title=f"Counts heatmap for {algorithm} - lowspeed_highrom",
        #              xlabels=["shortest", "short", "medium", "long", "longest"],
        #              ylabels=["closest", "close", "medium", "far", "farthest"])
        # plt.show()

    # def _ideal_distribution_for_profile(profile_name, mc_per_cell=300, target_prob=0.6, variability=0.25, total_trials=200):
    #     patient_params = PATIENT_PROFILES[profile_name]
    #     patient_seed = int(patient_params.get("seed", 7))

    #     ideal_mod = _load_ideal_distribution_module()
    #     estimate_true_phit_matrix = ideal_mod["estimate_true_phit_matrix"]
    #     make_ideal_distribution = ideal_mod["make_ideal_distribution"]
    #     PatientModel = _load_patient_model()

    #     class _ProfilePatientModel(PatientModel):
    #         _profile_params = patient_params

    #         def __init__(self, *args, **kwargs):
    #             params = dict(self._profile_params)
    #             params.setdefault("seed", kwargs.get("seed", 7))
    #             if "v_sigma0" not in params and "v_sigma" in kwargs:
    #                 params["v_sigma0"] = kwargs["v_sigma"]
    #             params.update({k: v for k, v in kwargs.items() if k in {"seed"}})
    #             super().__init__(**params)

    #     estimate_true_phit_matrix.__globals__["PatientModel"] = _ProfilePatientModel
    #     estimate_true_phit_matrix.__globals__["np"] = np
    #     estimate_true_phit_matrix.__globals__["distance_level_from_patient_bins"] = ideal_mod["distance_level_from_patient_bins"]
    #     estimate_true_phit_matrix.__globals__["D_BINS"] = ideal_mod["D_BINS"]
    #     estimate_true_phit_matrix.__globals__["T_BINS"] = ideal_mod["T_BINS"]

    #     phit_true = estimate_true_phit_matrix(
    #         patient_seed=patient_seed,
    #         patient_speed_sd=patient_params.get("v_sigma0", 0.04),
    #         mc_per_cell=mc_per_cell,
    #     )
    #     ideal_dist = make_ideal_distribution(
    #         phit_true,
    #         target_prob=target_prob,
    #         variability=variability,
    #         total_trials=total_trials,
    #     )
    #     return ideal_dist

    # def _bin25(d, t, d_min, d_max, t_min, t_max):
    #     u_d = (d - d_min) / (d_max - d_min + 1e-12)
    #     u_t = (t - t_min) / (t_max - t_min + 1e-12)
    #     if u_d < 0.2:
    #         i = 0
    #     elif u_d < 0.4:
    #         i = 1
    #     elif u_d < 0.6:
    #         i = 2
    #     elif u_d < 0.8:
    #         i = 3
    #     else:
    #         i = 4
    #     if u_t < 0.2:
    #         j = 0
    #     elif u_t < 0.4:
    #         j = 1
    #     elif u_t < 0.6:
    #         j = 2
    #     elif u_t < 0.8:
    #         j = 3
    #     else:
    #         j = 4
    #     return i, j

    # def _counts_from_hist(hist, d_min, d_max, t_min, t_max):
    #     counts = np.zeros((5, 5), dtype=int)
    #     for d, t in zip(hist["d"], hist["t"]):
    #         i, j = _bin25(float(d), float(t), d_min, d_max, t_min, t_max)
    #         counts[i, j] += 1
    #     return counts

    # ideal_mod = _load_ideal_distribution_module()
    # d_min = float(ideal_mod["D_MIN"])
    # d_max = float(ideal_mod["D_MAX"])
    # t_min = float(ideal_mod["T_MIN"])
    # t_max = float(ideal_mod["T_MAX"])

    # alg_names = list(ALGORITHMS)
    # profile_names = list(PATIENT_PROFILES.keys())
    # diff_matrix = np.zeros((len(alg_names), len(profile_names)), dtype=float)

    # for i, alg in enumerate(alg_names):
    #     for j, profile in enumerate(profile_names):
    #         result = run_algorithm(alg, patient_profile=profile)
    #         if isinstance(result, tuple) and len(result) == 2:
    #             hist, counts = result
    #         elif isinstance(result, dict):
    #             hist = result.get("hist", {})
    #             counts = _counts_from_hist(hist, d_min, d_max, t_min, t_max)
    #         else:
    #             raise TypeError(f"Unexpected return type from {alg}: {type(result)}")

    #         ideal_dist = _ideal_distribution_for_profile(profile)
    #         diff_matrix[i, j] = float(np.abs(counts - ideal_dist).sum())

    # plot_heatmap = _load_visualization_module().plot_heatmap
    # plot_heatmap(
    #     diff_matrix,
    #     title="Total absolute difference vs ideal distribution",
    #     xlabels=profile_names,
    #     ylabels=alg_names,
    #     annotate=True,
    # )
    # plt.show()