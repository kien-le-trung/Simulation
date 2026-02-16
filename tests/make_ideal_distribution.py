import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# -----------------------------
# Load your patient simulator
# -----------------------------
PATIENT_SIM_PATH = "patients/patient_simulation_v4.py"  # adjust if needed

spec = importlib.util.spec_from_file_location("patient_simulation_v4", PATIENT_SIM_PATH)
patient_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patient_mod)
PatientModel = patient_mod.PatientModel

# -----------------------------
# Discretize d, t into 5 levels
# -----------------------------
D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0

D_LEVELS = np.linspace(D_MIN, D_MAX, 5)
T_LEVELS = np.linspace(T_MIN, T_MAX, 5)
D_BINS = np.linspace(D_MIN, D_MAX, 6)
T_BINS = np.linspace(T_MIN, T_MAX, 6)

ACTIONS = [(i, j) for i in range(5) for j in range(5)]  # (d_idx, t_idx)
N_DIRECTIONS = 8

def direction_to_bins(direction: int) -> tuple[int, int]:
    d = int(np.clip(direction, 0, N_DIRECTIONS - 1))
    az = d % 4
    el = d // 4
    return az, el


def action_to_dt(a):
    i, j = a
    return float(D_LEVELS[i]), float(T_LEVELS[j])

# -----------------------------
# Map continuous d_sys to patient distance_level
# -----------------------------
def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    candidates = np.where(patient.d_means <= d_sys)[0]
    if len(candidates) == 0:
        return 0
    return int(candidates[-1])

# -----------------------------
# “Ideal” matrix (Monte Carlo)
# -----------------------------
def estimate_true_phit_matrix(patient_seed=7,
                              patient_speed = 0.08,
                              patient_speed_sd = 0.04,
                              k_d_decay = 0,
                              v_sigma_growth = 0.03,
                              spatial_strength_map = None,
                              mc_per_cell=300):
    rng = np.random.default_rng(123)
    phit = np.zeros((5, 5), dtype=float)

    patient_kwargs = dict(
        seed=patient_seed,
        k_d0_per_sec=patient_speed,
        v_sigma0=patient_speed_sd,
        k_d_decay=k_d_decay,
        v_sigma_growth=v_sigma_growth,
    )
    if spatial_strength_map is not None:
        patient_kwargs["spatial_strength_map"] = spatial_strength_map

    for i in range(5):
        for j in range(5):
            hits = 0
            patient = PatientModel(**patient_kwargs)
            d_low, d_high = float(D_BINS[i]), float(D_BINS[i + 1])
            t_low, t_high = float(T_BINS[j]), float(T_BINS[j + 1])

            prev_hit = True

            for _ in range(mc_per_cell):
                d = float(rng.uniform(d_low, d_high))
                t = float(rng.uniform(t_low, t_high))
                d_sys, t_sys = (d, t)
                lvl = distance_level_from_patient_bins(patient, d_sys)
                out = patient.sample_trial(t_sys=t_sys, d_sys=d_sys, distance_level=lvl, previous_hit=prev_hit)
                h = bool(out["hit"])
                hits += int(h)
                prev_hit = h
            phit[i, j] = hits / mc_per_cell

    return phit

def estimate_true_phit_tensor(patient_seed=7,
                              patient_speed = 0.20,
                              patient_speed_sd = 0.04,
                              patient_speed_sd_growth = 0.03,
                              k_d_decay = 0,
                              spatial_strength_map = None,
                              mc_per_cell=200):
    rng = np.random.default_rng(123)
    phit = np.zeros((5, 5, N_DIRECTIONS), dtype=float)

    patient_kwargs = dict(
        seed=patient_seed,
        k_d0_per_sec=patient_speed,
        v_sigma0=patient_speed_sd,
        v_sigma_growth=patient_speed_sd_growth,
        k_d_decay=k_d_decay,
    )
    if spatial_strength_map is not None:
        patient_kwargs["spatial_strength_map"] = spatial_strength_map

    for i in range(5):
        for j in range(5):
            for ddir in range(N_DIRECTIONS):
                hits = 0
                patient = PatientModel(**patient_kwargs)
                d_low, d_high = float(D_BINS[i]), float(D_BINS[i + 1])
                t_low, t_high = float(T_BINS[j]), float(T_BINS[j + 1])

                prev_hit = True

                for _ in range(mc_per_cell):
                    d = float(rng.uniform(d_low, d_high))
                    t = float(rng.uniform(t_low, t_high))
                    d_sys, t_sys = (d, t)
                    lvl = distance_level_from_patient_bins(patient, d_sys)
                    out = patient.sample_trial(
                        t_sys=t_sys,
                        d_sys=d_sys,
                        distance_level=lvl,
                        previous_hit=prev_hit,
                        direction_bin=ddir,
                    )
                    h = bool(out["hit"])
                    hits += int(h)
                    prev_hit = h
                phit[i, j, ddir] = hits / mc_per_cell

    return phit

# Gausisian ideal distribution (centered at target_prob with controllable spread)
def make_ideal_distribution(phit_true, 
                            target_prob=0.6, 
                            variability=0.2, 
                            total_trials=200):
    w = np.exp(-((phit_true - target_prob) / variability) ** 2)
    w /= w.sum()
    w *= total_trials
    w = np.round(w).astype(int)
    return w

phit_true = estimate_true_phit_matrix(patient_seed=7, mc_per_cell=1000, patient_speed=0.08, patient_speed_sd=0.04)
ideal_dist = make_ideal_distribution(phit_true, target_prob=0.6, variability=0.25, total_trials=200)

# heatmaps
xlabels = ["shortest", "short", "medium", "long", "longest"]  # time bins
ylabels = ["closest", "close", "medium", "far", "farthest"]   # distance bins

def plot_heatmap(mat, title, xlabels, ylabels, annotate=True):
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat.dtype == int:
                    txt = str(mat[i, j])
                else:
                    txt = f"{mat[i, j]:.2f}"
                plt.text(j, i, txt, ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    
def plot_patient_spatial_ability(
    patient: PatientModel,
    title="Patient spatial ability (velocity multiplier)",
    azimuth_labels=None,
    elevation_labels=None,
    annotate=True,
):
    if not hasattr(patient, "spatial_strength_map"):
        raise ValueError("Patient model has no spatial_strength_map attribute.")

    mat = np.array(patient.spatial_strength_map, dtype=float)
    if mat.shape == (8,):
        mat = mat.reshape(2, 4)
    if mat.shape != (2, 4):
        raise ValueError(f"Expected spatial_strength_map shape (8,) or (2, 4); got {mat.shape}.")

    if azimuth_labels is None:
        azimuth_labels = ["left", "front_left", "front_right", "right"]
    if elevation_labels is None:
        elevation_labels = ["lower", "upper"]

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(azimuth_labels)), azimuth_labels)
    plt.yticks(range(len(elevation_labels)), elevation_labels)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(label="Velocity multiplier")
    plt.tight_layout()
    plt.show()

def plot_phit_dt(phit_tensor, direction=None, reduce="mean"):
    if direction is None:
        if reduce == "mean":
            mat = np.mean(phit_tensor, axis=2)
        elif reduce == "min":
            mat = np.min(phit_tensor, axis=2)
        elif reduce == "max":
            mat = np.max(phit_tensor, axis=2)
        else:
            raise ValueError("reduce must be one of: mean, min, max")
        title = f"P(hit) vs (d,t), dir={reduce}"
    else:
        ddir = int(np.clip(direction, 0, N_DIRECTIONS - 1))
        mat = phit_tensor[:, :, ddir]
        title = f"P(hit) vs (d,t), dir={ddir}"
    plot_heatmap(mat, title, xlabels, ylabels, annotate=True)

def plot_phit_d_dir(phit_tensor, t_idx=None, reduce="mean"):
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

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xticks(range(N_DIRECTIONS), [f"dir{d}" for d in range(N_DIRECTIONS)])
    plt.yticks(range(5), ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

phit_true = estimate_true_phit_tensor(patient_seed=7, 
                                      mc_per_cell=500, 
                                      patient_speed=0.10,
                                      patient_speed_sd=0.04,
                                      k_d_decay=0,
                                      spatial_strength_map=None)
plot_phit_d_dir(phit_true, reduce="mean")