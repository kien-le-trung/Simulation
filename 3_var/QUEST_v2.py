# quest_plus_dt.py
import importlib.util
import math
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v3.py"
patient_sim = _load_module_from_path("patients.patient_simulation_v3", PATIENT_SIM_PATH)
PatientModel = patient_sim.PatientModel


D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0

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

def level5(x, xmin, xmax):
    """
    Map x in [xmin,xmax] -> {0,1,2,3,4} by fifths.
    Bin 0 = shortest/closest (0-20th percentile)
    Bin 1 = short/close (20-40th percentile)
    Bin 2 = medium (40-60th percentile)
    Bin 3 = long/far (60-80th percentile)
    Bin 4 = longest/farthest (80-100th percentile)
    """
    u = (x - xmin) / (xmax - xmin + 1e-12)
    if u < 0.2: return 0
    if u < 0.4: return 1
    if u < 0.6: return 2
    if u < 0.8: return 3
    return 4

def bin25(d, t):
    return (level5(d, D_MIN, D_MAX), level5(t, T_MIN, T_MAX))  # (dist_level, time_level)

# ----------------------------
# QUEST+ main
# ----------------------------
def run_quest_plus_dt(
    patient: PatientModel,
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

    counts = np.zeros((5, 5), dtype=int)

    # default grids (keep modest to avoid huge compute)
    if d_grid is None:
        d_grid = np.round(np.arange(D_MIN, D_MAX + 0.01, 0.05), 4)   # 15 values
    if t_grid is None:
        t_grid = np.round(np.arange(T_MIN, T_MAX + 0.01, 0.25), 4)   # 25 values

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

        i, j = bin25(d_sys, t_sys)
        counts[i, j] += 1

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

    return hist, counts