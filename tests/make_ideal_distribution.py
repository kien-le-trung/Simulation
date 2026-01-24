import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# -----------------------------
# Load your patient simulator
# -----------------------------
PATIENT_SIM_PATH = "patient_simulation_v3.py"  # adjust if needed

spec = importlib.util.spec_from_file_location("patient_simulation_v3", PATIENT_SIM_PATH)
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
                              mc_per_cell=300):
    rng = np.random.default_rng(123)
    phit = np.zeros((5, 5), dtype=float)

    for i in range(5):
        for j in range(5):
            hits = 0
            patient = PatientModel(seed=patient_seed,
                                   k_d0_per_sec=0.40,
                                   v_sigma=patient_speed_sd)
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

# Gausisian ideal distribution (centered at target_prob with controllable spread)
def make_ideal_distribution(phit_true, 
                            target_prob=0.6, 
                            variability=0.15, 
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

    

plot_heatmap(phit_true, "True P(hit) per cell (Monte Carlo)", xlabels, ylabels, annotate=True)
plot_heatmap(ideal_dist, "Ideal distribution over (d,t) cells", xlabels, ylabels, annotate=True)