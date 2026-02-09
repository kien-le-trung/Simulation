# rl_logreg_difficulty_index.py
# Difficulty-index + logistic regression + RL-style updates to index weights.
#
# Requires: numpy, matplotlib
# Uses your patient simulator: patient_simulation_v4.py

import importlib.util
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel

# -----------------------------
# Discretize d, t into 5 levels
# -----------------------------
D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0

D_LEVELS = np.linspace(D_MIN, D_MAX, 5)
T_LEVELS = np.linspace(T_MIN, T_MAX, 5)

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
# Jitter (harder/easier execution)
# -----------------------------
def apply_jitter(d, t, rng, p_jitter=0.7, d_sigma=0.06, t_sigma=0.10):
    d_sys, t_sys = d, t
    if rng.random() < p_jitter:
        if rng.random() < 0.5:
            eps = rng.normal(0.0, d_sigma)
            d_sys = float(np.clip(d * (1.0 + eps), D_MIN, D_MAX))
        else:
            eps = rng.normal(0.0, t_sigma)
            t_sys = float(np.clip(t * (1.0 + eps), T_MIN, T_MAX))
    return d_sys, t_sys


# -----------------------------
# Difficulty index and model
# -----------------------------
def sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_d(d):
    return (d - D_MIN) / (D_MAX - D_MIN + 1e-12)


def hard_time_feature(t):
    # bigger means harder: shortest time => 1, longest time => 0
    return 1.0 - (t - T_MIN) / (T_MAX - T_MIN + 1e-12)


INDEX_W_D = 1.0
INDEX_W_T = 1.0
INDEX_W_DIR = 1.0
N_DIRECTIONS = 8


def normalize_dir(direction):
    return float(direction) / max(1.0, (N_DIRECTIONS - 1))


class DifficultyIndexLogReg:
    """
    p_hat = sigmoid(beta0 + beta1 * I)
    I = INDEX_W_D * d_norm + INDEX_W_T * hard_time + INDEX_W_DIR * dir_norm

    beta updated by supervised log-loss gradient (hit/miss).
    """
    def __init__(
        self,
        rng,
        beta0=0.0,
        beta1=1.0,
        lr_beta=0.25,
        p_star=0.65,
        lam_p=0.55,
    ):
        self.rng = rng
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)

        self.w_d = float(INDEX_W_D)
        self.w_t = float(INDEX_W_T)
        self.w_dir = float(INDEX_W_DIR)

        self.lr_beta = float(lr_beta)

        self.p_star = float(p_star)
        self.lam_p = float(lam_p)

        self._seen = set()  # track if we've seen both classes for stable behavior

    def index(self, d, t, direction):
        dn = normalize_d(d)
        ht = hard_time_feature(t)
        dr = normalize_dir(direction)
        return self.w_d * dn + self.w_t * ht + self.w_dir * dr, dn, ht, dr

    def predict_p(self, d, t, direction):
        # before seeing both classes, be maximally uncertain
        if len(self._seen) < 2:
            return 0.5
        I, _, _, _ = self.index(d, t, direction)
        return float(sigmoid(self.beta0 + self.beta1 * I))

    def uncertainty(self, d, t, direction):
        p = self.predict_p(d, t, direction)
        return 1.0 - 2.0 * abs(p - 0.5)  # in [0,1]

    def update(self, hit, d_exec, t_exec, dir_exec, d_intended, t_intended, dir_intended):
        """
        Supervised update for beta from executed (d_exec, t_exec) and hit label.
        """
        y = 1.0 if hit else 0.0
        self._seen.add(int(y))

        # logistic regression update on executed stimulus
        I_exec, _, _, _ = self.index(d_exec, t_exec, dir_exec)
        p_exec = sigmoid(self.beta0 + self.beta1 * I_exec)

        # gradient of log-loss: (p - y)
        g = (p_exec - y)
        # d/d beta0 = g
        # d/d beta1 = g * I
        self.beta0 -= self.lr_beta * g
        self.beta1 -= self.lr_beta * g * I_exec

        # keep beta1 positive-ish so larger I => harder => lower p (or higher, depending patient)
        # We don't force sign, but stabilize extreme drift:
        self.beta1 = float(np.clip(self.beta1, -8.0, 8.0))
        self.beta0 = float(np.clip(self.beta0, -12.0, 12.0))

        return float(p_exec), float(I_exec), None


# -----------------------------
# Controller run
# -----------------------------
def run_controller(
    patient: PatientModel,
    n_trials=200,
    seed=7,
    p_star=0.65,
    explore_prob=0.40,  # active-learning exploration near p=0.5
    p_jitter=0.75,
):
    rng = np.random.default_rng(seed)

    model = DifficultyIndexLogReg(
        rng=rng,
        beta0=0.0,
        beta1=1.0,
        lr_beta=0.30,
        p_star=p_star,
        lam_p=0.55,
    )

    counts = np.zeros((5, 5), dtype=int)

    hist = {
        "a_i": [], "a_j": [],
        "d": [], "t": [],
        "d_sys": [], "t_sys": [],
        "direction": [],
        "p_pred_intended": [],
        "p_pred_exec": [],
        "uncert": [],
        "hit": [],
        "rolling_hit": [],
        "beta0": [], "beta1": [],
        "w_d": [], "w_t": [], "w_dir": [],
        "index_exec": [],
    }

    prev_hit = True
    rolling_window = 20
    hit_list = []

    for k in range(n_trials):
        dir_samples = rng.beta(patient.spatial_success_alpha,
                               patient.spatial_success_beta)
        direction = int(np.argmin(np.abs(dir_samples - p_star)))
        # compute predicted p and uncertainty for all 25 actions (intended grid points)
        p_hat = np.zeros((5, 5), dtype=float)
        uncert = np.zeros((5, 5), dtype=float)
        for i in range(5):
            for j in range(5):
                d, t = float(D_LEVELS[i]), float(T_LEVELS[j])
                p_hat[i, j] = model.predict_p(d, t, direction)
                uncert[i, j] = 1.0 - 2.0 * abs(p_hat[i, j] - 0.5)

        # choose action
        if rng.random() < explore_prob:
            # Active learning: pick highest uncertainty (closest to p=0.5)
            # tie-break: prefer slightly harder (farther d and shorter t)
            best = None
            best_score = -1e18
            for i in range(5):
                for j in range(5):
                    d, t = float(D_LEVELS[i]), float(T_LEVELS[j])
                    dn = normalize_d(d)
                    ht = hard_time_feature(t)
                    score = uncert[i, j] #+ 0.08 * dn + 0.06 * ht
                    if score > best_score:
                        best_score = score
                        best = (i, j)
            a = best
        else:
            # Exploit: choose actions whose predicted p is close to p_star, and nudged harder
            # This is "policy from model" instead of tabular Q.
            score = -((p_hat - p_star) ** 2)
            score += 0.08 * (D_LEVELS[:, None] - D_MIN) / (D_MAX - D_MIN + 1e-12)
            score += 0.05 * hard_time_feature(T_LEVELS[None, :])
            # softmax sample
            z = score.reshape(-1)
            z = z - np.max(z)
            probs = np.exp(z / 0.12)
            probs = probs / probs.sum()
            idx = rng.choice(len(ACTIONS), p=probs)
            a = ACTIONS[idx]

        i, j = a
        d, t = action_to_dt(a)

        # apply jitter to executed system values
        d_sys, t_sys = apply_jitter(d, t, rng, p_jitter=p_jitter)

        # run patient trial
        lvl = distance_level_from_patient_bins(patient, d_sys)
        out = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=prev_hit,
            direction_bin=direction,
        )
        hit = bool(out["hit"])
        prev_hit = hit
        if hit:
            patient.spatial_success_alpha[direction] += 1.0
        else:
            patient.spatial_success_beta[direction] += 1.0

        # update model (beta on executed, w on intended via shaped reward)
        p_exec_before = model.predict_p(d_sys, t_sys, direction)
        p_int_before = model.predict_p(d, t, direction)
        p_exec_after, I_exec, p_int_after = model.update(
            hit, d_sys, t_sys, direction, d, t, direction
        )

        counts[i, j] += 1

        # rolling hit rate
        hit_list.append(int(hit))
        if len(hit_list) >= rolling_window:
            roll = float(np.mean(hit_list[-rolling_window:]))
        else:
            roll = float(np.mean(hit_list))

        # log
        hist["a_i"].append(i)
        hist["a_j"].append(j)
        hist["d"].append(d)
        hist["t"].append(t)
        hist["d_sys"].append(d_sys)
        hist["t_sys"].append(t_sys)
        hist["direction"].append(int(direction))
        hist["p_pred_intended"].append(float(p_int_before))
        hist["p_pred_exec"].append(float(p_exec_before))
        hist["uncert"].append(float(uncert[i, j]))
        hist["hit"].append(int(hit))
        hist["rolling_hit"].append(roll)
        hist["beta0"].append(float(model.beta0))
        hist["beta1"].append(float(model.beta1))
        hist["w_d"].append(float(model.w_d))
        hist["w_t"].append(float(model.w_t))
        hist["w_dir"].append(float(model.w_dir))
        hist["index_exec"].append(float(I_exec))

    return hist, counts, patient


