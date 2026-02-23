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
# Candidate grid for (d, t, dir)
# -----------------------------
D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0

D_STEP = 0.05
T_STEP = 0.5

D_GRID = np.round(np.arange(D_MIN, D_MAX + 1e-9, D_STEP), 4)
T_GRID = np.round(np.arange(T_MIN, T_MAX + 1e-9, T_STEP), 4)
CANDIDATES = [(float(d), float(t), int(direction)) for d in D_GRID for t in T_GRID for direction in range(9)]


def level5(x, xmin, xmax):
    u = (x - xmin) / (xmax - xmin + 1e-12)
    if u < 0.2: return 0
    if u < 0.4: return 1
    if u < 0.6: return 2
    if u < 0.8: return 3
    return 4


def bin25(d, t):
    return (level5(d, D_MIN, D_MAX), level5(t, T_MIN, T_MAX))


def pick_with_tiebreak(candidates, score_fn, rng, maximize=False, atol=1e-12):
    scores = np.array([float(score_fn(c)) for c in candidates], dtype=float)
    best_score = float(np.max(scores) if maximize else np.min(scores))
    tie_idx = np.where(np.isclose(scores, best_score, atol=atol, rtol=0.0))[0]
    chosen = int(rng.choice(tie_idx))
    return candidates[chosen]


# -----------------------------
# Map continuous d_sys to patient distance_level
# -----------------------------
def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    candidates = np.where(patient.d_means <= d_sys)[0]
    if len(candidates) == 0:
        return 0
    return int(candidates[-1])


def apply_calibration_priors(patient: PatientModel, calibration_result: dict | None):
    if not calibration_result:
        return
    per_direction = calibration_result.get("per_direction", {})
    for direction, stats in per_direction.items():
        idx = int(np.clip(direction, 0, 8))
        n_reached = float(stats.get("n_reached", 0))
        n_censored = float(stats.get("n_censored", 0))
        patient.spatial_success_alpha[idx] += n_reached
        patient.spatial_success_beta[idx] += n_censored


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
N_DIRECTIONS = 9


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
        p_star=0.70,
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
    p_star=0.70,
    p_tol=0.05,
    explore_prob=0.10,  # active-learning exploration near p=0.5
    p_jitter=0.30,
    calibration=True,
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

    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        for trial in calibration_result.get("trials", []):
            d_exec = float(trial.get("d_sys", D_MIN))
            t_exec = float(np.clip(trial.get("t_cap", T_MAX), T_MIN, T_MAX))
            direction = int(np.clip(trial.get("direction_bin", 0), 0, N_DIRECTIONS - 1))
            hit = bool(trial.get("hit", False))
            model.update(hit, d_exec, t_exec, direction, d_exec, t_exec, direction)

    for k in range(n_trials):
        # score full candidate grid, then keep only near-target p(hit)
        scored = []
        for d, t, direction in CANDIDATES:
            p_pred = model.predict_p(d, t, direction)
            uncert = 1.0 - 2.0 * abs(p_pred - 0.5)
            scored.append((d, t, direction, p_pred, uncert))

        near_target = [x for x in scored if abs(x[3] - p_star) <= p_tol]
        pool = near_target if len(near_target) > 0 else scored

        # choose candidate with exploration probability
        if rng.random() < explore_prob:
            # exploration: pick highest-uncertainty candidate in pool
            d, t, direction, p_pred_intended, uncert_val = pick_with_tiebreak(
                pool, score_fn=lambda x: x[4], rng=rng, maximize=True
            )
        else:
            # exploitation: best p(hit) match to target in pool
            d, t, direction, p_pred_intended, uncert_val = pick_with_tiebreak(
                pool, score_fn=lambda x: abs(x[3] - p_star), rng=rng, maximize=False
            )

        i, j = bin25(d, t)

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
        p_int_before = p_pred_intended
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
        hist["uncert"].append(float(uncert_val))
        hist["hit"].append(int(hit))
        hist["rolling_hit"].append(roll)
        hist["beta0"].append(float(model.beta0))
        hist["beta1"].append(float(model.beta1))
        hist["w_d"].append(float(model.w_d))
        hist["w_t"].append(float(model.w_t))
        hist["w_dir"].append(float(model.w_dir))
        hist["index_exec"].append(float(I_exec))

    return hist, counts, patient


