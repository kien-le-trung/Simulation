# quest_plus_dt.py
import numpy as np
import matplotlib.pyplot as plt
import math
import importlib.util

# ----------------------------
# Load your patient simulator
# ----------------------------
PATIENT_SIM_PATH = "patient_simulation_v3.py"  # adjust if needed
PATIENT_SPEED = 0.08 # m/s
PATIENT_SPEED_SD = 0.04 # m/s

spec = importlib.util.spec_from_file_location("patient_simulation_v3", PATIENT_SIM_PATH)
patient_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patient_sim)
PatientModel = patient_sim.PatientModel

matrix_spec = importlib.util.spec_from_file_location("make_ideal_distribution", "tests/make_ideal_distribution.py")
matrix_mod = importlib.util.module_from_spec(matrix_spec)
matrix_spec.loader.exec_module(matrix_mod)
true_p_hit = matrix_mod.estimate_true_phit_matrix(patient_seed=7, 
                                                  mc_per_cell=1000, 
                                                  patient_speed=PATIENT_SPEED, 
                                                  patient_speed_sd=PATIENT_SPEED_SD)
ideal_matrix = matrix_mod.make_ideal_distribution(true_p_hit, target_prob=0.6, variability=0.25, total_trials=200)


# ----------------------------
# QUEST+ helpers
# ----------------------------
def sigmoid(x):
    # stable sigmoid
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def entropy(posterior, eps=1e-12):
    """Shannon entropy of a discrete posterior array."""
    p = posterior.ravel().astype(float)
    p = p / (p.sum() + eps)
    p = p[p > 0]
    return -np.sum(p * np.log(p + eps))


def normalize(p, eps=1e-12):
    s = np.sum(p)
    if s <= eps:
        return np.full_like(p, 1.0 / p.size)
    return p / s


def cartesian_params(v_grid, r_grid, beta_grid):
    """Return parameter lattice arrays (V,R,B) broadcastable to likelihood computations."""
    V = v_grid[:, None, None]
    R = r_grid[None, :, None]
    B = beta_grid[None, None, :]
    return V, R, B


def p_hit_model(d, t, V, R, B):
    """
    Psychometric model (NO lapse):
        p_hit = sigmoid( beta * (t - r - d/v) )
    """
    # avoid division by 0
    V_safe = np.maximum(V, 1e-6)
    margin = (t - R - d / V_safe)
    return sigmoid(B * margin)


def expected_entropy_for_x(d, t, posterior, V, R, B, eps=1e-12):
    """
    Compute expected posterior entropy if we present stimulus x=(d,t).
    Two outcomes: hit=1, miss=0.
    """
    # likelihood for hit/miss at each parameter point
    p_hit = p_hit_model(d, t, V, R, B)          # shape (nv,nr,nb)
    p_miss = 1.0 - p_hit

    # predictive probabilities under current posterior
    post = posterior
    p_hit_pred = float(np.sum(post * p_hit))
    p_miss_pred = 1.0 - p_hit_pred

    # posterior updates for each outcome (unnormalized)
    post_hit = normalize(post * p_hit, eps=eps)
    post_miss = normalize(post * p_miss, eps=eps)

    # expected entropy
    H_hit = entropy(post_hit, eps=eps)
    H_miss = entropy(post_miss, eps=eps)

    EH = p_hit_pred * H_hit + p_miss_pred * H_miss
    return EH, p_hit_pred


def map_distance_level(patient: PatientModel, d_sys: float) -> int:
    """
    Map continuous d_sys to the PatientModel distance_level used by sample_trial.
    This mirrors the common rule: choose the largest bin whose mean distance <= d_sys.
    """
    d_means = np.asarray(patient.d_means, dtype=float)
    idxs = np.where(d_means <= d_sys)[0]
    if len(idxs) == 0:
        return 0
    return int(idxs[-1])


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def level5(x: float, xmin: float, xmax: float) -> int:
    """
    Map x in [xmin,xmax] -> {0,1,2,3,4} by fifths.
    Bin 0 is smallest value (closest/shortest), Bin 4 is largest (farthest/longest).
    """
    if xmax <= xmin:
        return 0
    r = (x - xmin) / (xmax - xmin)
    r = clamp(r, 0.0, 1.0)
    return int(min(4, math.floor(5 * r + 1e-12)))


def bin25(d: float, t: float, d_min: float, d_max: float, t_min: float, t_max: float):
    """
    Returns (dist_bin, time_bin) in {0..4}x{0..4}.
    dist_bin 0=closest, 4=farthest; time_bin 0=shortest, 4=longest.
    """
    i = level5(d, d_min, d_max)
    j = level5(t, t_min, t_max)
    return i, j


def count_trials_per_bin(
    d_list,
    t_list,
    d_min: float,
    d_max: float,
    t_min: float,
    t_max: float,
):
    """
    Count trials for each (d,t) bin in a 5x5 grid.
    Rows=distance bins (closest->farthest), cols=time bins (shortest->longest).
    """
    counts = np.zeros((5, 5), dtype=int)
    for d, t in zip(d_list, t_list):
        i, j = bin25(float(d), float(t), d_min, d_max, t_min, t_max)
        counts[i, j] += 1
    return counts


BIN_NAMES_5 = ["closest/shortest", "close/short", "medium", "far/long", "farthest/longest"]


# ----------------------------
# QUEST+ main
# ----------------------------
def run_quest_plus_dt(
    n_trials=120,
    p_target=0.65,
    p_tol=0.08,
    seed=7,
    # stimulus grid (d,t)
    d_grid=None,
    t_grid=None,
    # parameter grid (v,r,beta)
    v_grid=None,
    r_grid=None,
    beta_grid=None,
):
    """
    QUEST+ loop:
      - maintain posterior over (v,r,beta)
      - choose (d,t) minimizing expected entropy, but only among points whose
        posterior-predictive P(hit) is near p_target (within p_tol). Fallback to global min EH.
      - query patient simulator for outcome, update posterior.

    Returns: dict with posterior, grids, and trial history.
    """
    rng = np.random.default_rng(seed)
    patient = PatientModel(seed=seed,
                           v_mean=PATIENT_SPEED,
                           v_sigma=PATIENT_SPEED_SD)

    # default grids (keep modest to avoid huge compute)
    if d_grid is None:
        d_grid = np.round(np.arange(0.10, 0.81, 0.05), 4)   # 15 values
    if t_grid is None:
        t_grid = np.round(np.arange(1.00, 7.01, 0.25), 4)   # 25 values

    if v_grid is None:
        v_grid = np.round(np.linspace(0.06, 0.22, 17), 4)   # speed (m/s)
    if r_grid is None:
        r_grid = np.round(np.linspace(0.10, 0.60, 11), 4)   # reaction/latency (s)
    if beta_grid is None:
        beta_grid = np.round(np.linspace(2.0, 10.0, 9), 4)  # slope

    # parameter lattice
    V, R, B = cartesian_params(v_grid, r_grid, beta_grid)   # broadcastable
    posterior = np.ones((len(v_grid), len(r_grid), len(beta_grid)), dtype=float)
    posterior = normalize(posterior)

    # history
    hist = {
        "d": [],
        "t": [],
        "p_pred": [],
        "EH": [],
        "hit": [],
        "t_pat": [],
        "d_pat": [],
        "time_ratio": [],
        "dist_ratio": [],
        "H_post": [],
    }

    prev_hit = True

    # prebuild stimulus list
    stimuli = [(float(d), float(t)) for d in d_grid for t in t_grid]

    for k in range(n_trials):
        # 1) Evaluate candidates
        cand = []
        for (d, t) in stimuli:
            EH, p_pred = expected_entropy_for_x(d, t, posterior, V, R, B)
            cand.append((EH, p_pred, d, t))
        
        # Choose points with the highest uncertainty (highest expected entropy) 20% of the time
        if k % 5 == 0:
            near = cand
        else:
            # 2) Filter to near-target p(hit)
            near = [x for x in cand if abs(x[1] - p_target) <= p_tol]

        # pick best by min expected entropy (fallback to global min)
        chosen = min(near, key=lambda x: x[0]) if len(near) > 0 else min(cand, key=lambda x: x[0])
        EH_best, p_pred_best, d_sys, t_sys = chosen

        # 3) Query patient simulator
        lvl = map_distance_level(patient, d_sys)
        out = patient.sample_trial(t_sys=t_sys, d_sys=d_sys, distance_level=lvl, previous_hit=prev_hit)
        hit = bool(out["hit"])
        prev_hit = hit

        # 4) Bayesian update
        # likelihood at chosen stimulus
        p_hit = p_hit_model(d_sys, t_sys, V, R, B)
        like = p_hit if hit else (1.0 - p_hit)
        posterior = normalize(posterior * like)

        # 5) Log
        hist["d"].append(d_sys)
        hist["t"].append(t_sys)
        hist["p_pred"].append(p_pred_best)
        hist["EH"].append(EH_best)
        hist["hit"].append(int(hit))
        hist["t_pat"].append(float(out["t_pat"]))
        hist["d_pat"].append(float(out["d_pat"]))
        hist["time_ratio"].append(float(out["time_ratio"]))
        hist["dist_ratio"].append(float(out["dist_ratio"]))
        hist["H_post"].append(entropy(posterior))

    return {
        "posterior": posterior,
        "d_grid": d_grid,
        "t_grid": t_grid,
        "v_grid": v_grid,
        "r_grid": r_grid,
        "beta_grid": beta_grid,
        "hist": hist,
        "patient": patient,
    }


# ----------------------------
# Visualization
# ----------------------------
def posterior_marginals(result):
    post = result["posterior"]
    v_grid = result["v_grid"]
    r_grid = result["r_grid"]
    beta_grid = result["beta_grid"]

    p_v = post.sum(axis=(1, 2))
    p_r = post.sum(axis=(0, 2))
    p_b = post.sum(axis=(0, 1))

    return (v_grid, p_v), (r_grid, p_r), (beta_grid, p_b)


def predictive_p_hit_surface(result):
    """
    Compute posterior-predictive p_hit(d,t) over the (d,t) grid.
    """
    posterior = result["posterior"]
    d_grid = result["d_grid"]
    t_grid = result["t_grid"]
    v_grid = result["v_grid"]
    r_grid = result["r_grid"]
    beta_grid = result["beta_grid"]
    V, R, B = cartesian_params(v_grid, r_grid, beta_grid)

    P = np.zeros((len(d_grid), len(t_grid)), dtype=float)
    for i, d in enumerate(d_grid):
        for j, t in enumerate(t_grid):
            p_hit = p_hit_model(float(d), float(t), V, R, B)
            P[i, j] = float(np.sum(posterior * p_hit))
    return P


def rolling_mean(x, w=10):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_quest_plus_results(result, p_target=0.65):
    hist = result["hist"]
    d = np.array(hist["d"])
    t = np.array(hist["t"])
    hit = np.array(hist["hit"])
    p_pred = np.array(hist["p_pred"])
    H_post = np.array(hist["H_post"])

    # 1) trajectory of chosen (d,t)
    plt.figure()
    plt.scatter(d, t, s=14)
    plt.title("QUEST+ chosen (d,t) across trials")
    plt.xlabel("Distance d (m)")
    plt.ylabel("Time-to-live t (s)")
    plt.show()

    # 2) predicted p(hit) vs rolling hit-rate
    plt.figure()
    plt.plot(p_pred, label="posterior-predictive p(hit) at chosen (d,t)")
    rm = rolling_mean(hit, w=10)
    plt.plot(np.arange(len(rm)) + 10 - 1, rm, linestyle="--", label="rolling hit rate (w=10)")
    plt.axhline(p_target, linestyle=":", label=f"target p={p_target:.2f}")
    plt.ylim(-0.05, 1.05)
    plt.title("Targeting hit probability")
    plt.xlabel("Trial")
    plt.ylabel("Probability / rate")
    plt.legend()
    plt.show()

    # 3) posterior entropy over time
    plt.figure()
    plt.plot(H_post)
    plt.title("Posterior entropy over trials (should generally decrease)")
    plt.xlabel("Trial")
    plt.ylabel("Entropy")
    plt.show()

    # 4) posterior marginals
    (v_grid, p_v), (r_grid, p_r), (b_grid, p_b) = posterior_marginals(result)

    plt.figure()
    plt.plot(v_grid, p_v)
    plt.title("Posterior marginal: speed v")
    plt.xlabel("v (m/s)")
    plt.ylabel("Posterior mass")
    plt.show()

    plt.figure()
    plt.plot(r_grid, p_r)
    plt.title("Posterior marginal: reaction time r")
    plt.xlabel("r (s)")
    plt.ylabel("Posterior mass")
    plt.show()

    plt.figure()
    plt.plot(b_grid, p_b)
    plt.title("Posterior marginal: slope beta")
    plt.xlabel("beta")
    plt.ylabel("Posterior mass")
    plt.show()

    # 5) posterior-predictive p(hit) surface over (d,t), with sampled points overlay
    P = predictive_p_hit_surface(result)
    d_grid = result["d_grid"]
    t_grid = result["t_grid"]

    plt.figure()
    plt.imshow(
        P.T,
        origin="lower",
        aspect="auto",
        extent=[d_grid[0], d_grid[-1], t_grid[0], t_grid[-1]],
    )
    plt.colorbar(label="posterior-predictive p(hit)")
    plt.scatter(d, t, s=10, marker="o")
    plt.title("Posterior-predictive p(hit) over (d,t) + chosen samples")
    plt.xlabel("d (m)")
    plt.ylabel("t (s)")
    plt.show()

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

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    res = run_quest_plus_dt(
        n_trials=200,
        p_target=0.6,
        p_tol=0.1,
        seed=7,
    )
    # plot_quest_plus_results(res, p_target=0.6)

    hist = res["hist"]
    d_vals = hist["d"]
    t_vals = hist["t"]
    d_min, d_max = min(d_vals), max(d_vals)
    t_min, t_max = min(t_vals), max(t_vals)
    counts = count_trials_per_bin(d_vals, t_vals, d_min, d_max, t_min, t_max)
    plot_heatmap(
        counts,
        "QUEST+: Trial Counts per (d,t) Bin",
        BIN_NAMES_5,
        BIN_NAMES_5,
        annotate=True,
    )
    print(f"Absolute difference: {np.abs(counts - ideal_matrix).sum()}")