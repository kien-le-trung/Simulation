# quest_plus_dt.py
import numpy as np
import matplotlib.pyplot as plt
import math
import importlib.util

# ----------------------------
# Load your patient simulator
# ----------------------------
PATIENT_SIM_PATH = "patient_simulation_v2.py"  # adjust if needed

spec = importlib.util.spec_from_file_location("patient_simulation_v2", PATIENT_SIM_PATH)
patient_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patient_sim)
PatientModel = patient_sim.PatientModel


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


def cartesian_params(b0_grid, b1_grid, b2_grid):
    """Return parameter lattice arrays (B0,B1,B2) broadcastable to likelihood computations."""
    B0 = b0_grid[:, None, None]
    B1 = b1_grid[None, :, None]
    B2 = b2_grid[None, None, :]
    return B0, B1, B2


def p_hit_model(d, t, B0, B1, B2):
    """
    Psychometric model (NO lapse):
        p_hit = sigmoid(B0 + B1*d + B2*t)
    """
    return sigmoid(B0 + B1 * d + B2 * t)


def expected_entropy_for_x(d, t, posterior, B0, B1, B2, eps=1e-12):
    """
    Compute expected posterior entropy if we present stimulus x=(d,t).
    Two outcomes: hit=1, miss=0.
    """
    # likelihood for hit/miss at each parameter point
    p_hit = p_hit_model(d, t, B0, B1, B2)          # shape (nb0,nb1,nb2)
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
    # parameter grid (B0,B1,B2)
    b0_grid=None,
    b1_grid=None,
    b2_grid=None,
):
    """
    QUEST+ loop:
      - maintain posterior over (B0,B1,B2)
      - choose (d,t) minimizing expected entropy, but only among points whose
        posterior-predictive P(hit) is near p_target (within p_tol). Fallback to global min EH.
      - query patient simulator for outcome, update posterior.

    Returns: dict with posterior, grids, and trial history.
    """
    rng = np.random.default_rng(seed)
    patient = PatientModel(seed=seed)

    # default grids (keep modest to avoid huge compute)
    if d_grid is None:
        d_grid = np.round(np.arange(0.10, 0.81, 0.05), 4)   # 15 values
    if t_grid is None:
        t_grid = np.round(np.arange(1.00, 7.01, 0.25), 4)   # 25 values

    if b0_grid is None:
        b0_grid = np.round(np.linspace(-6.0, 6.0, 17), 4)   # intercept
    if b1_grid is None:
        b1_grid = np.round(np.linspace(-10.0, 10.0, 21), 4) # weight on d
    if b2_grid is None:
        b2_grid = np.round(np.linspace(-2.0, 2.0, 17), 4)   # weight on t

    # parameter lattice
    B0, B1, B2 = cartesian_params(b0_grid, b1_grid, b2_grid)   # broadcastable
    posterior = np.ones((len(b0_grid), len(b1_grid), len(b2_grid)), dtype=float)
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
            EH, p_pred = expected_entropy_for_x(d, t, posterior, B0, B1, B2)
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
        p_hit = p_hit_model(d_sys, t_sys, B0, B1, B2)
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
        "b0_grid": b0_grid,
        "b1_grid": b1_grid,
        "b2_grid": b2_grid,
        "hist": hist,
        "patient": patient,
    }


# ----------------------------
# Visualization
# ----------------------------
def posterior_marginals(result):
    post = result["posterior"]
    b0_grid = result["b0_grid"]
    b1_grid = result["b1_grid"]
    b2_grid = result["b2_grid"]

    p_b0 = post.sum(axis=(1, 2))
    p_b1 = post.sum(axis=(0, 2))
    p_b2 = post.sum(axis=(0, 1))

    return (b0_grid, p_b0), (b1_grid, p_b1), (b2_grid, p_b2)


def predictive_p_hit_surface(result):
    """
    Compute posterior-predictive p_hit(d,t) over the (d,t) grid.
    """
    posterior = result["posterior"]
    d_grid = result["d_grid"]
    t_grid = result["t_grid"]
    b0_grid = result["b0_grid"]
    b1_grid = result["b1_grid"]
    b2_grid = result["b2_grid"]
    B0, B1, B2 = cartesian_params(b0_grid, b1_grid, b2_grid)

    P = np.zeros((len(d_grid), len(t_grid)), dtype=float)
    for i, d in enumerate(d_grid):
        for j, t in enumerate(t_grid):
            p_hit = p_hit_model(float(d), float(t), B0, B1, B2)
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
    (b0_grid, p_b0), (b1_grid, p_b1), (b2_grid, p_b2) = posterior_marginals(result)

    plt.figure()
    plt.plot(b0_grid, p_b0)
    plt.title("Posterior marginal: B0")
    plt.xlabel("B0")
    plt.ylabel("Posterior mass")
    plt.show()

    plt.figure()
    plt.plot(b1_grid, p_b1)
    plt.title("Posterior marginal: B1")
    plt.xlabel("B1")
    plt.ylabel("Posterior mass")
    plt.show()

    plt.figure()
    plt.plot(b2_grid, p_b2)
    plt.title("Posterior marginal: B2")
    plt.xlabel("B2")
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
    plot_quest_plus_results(res, p_target=0.6)
