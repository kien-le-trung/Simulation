# rl_logreg_difficulty_index.py
# Difficulty-index + logistic regression + RL-style updates to index weights.
#
# Requires: numpy, matplotlib
# Uses your patient simulator: patient_simulation_v2.py  :contentReference[oaicite:0]{index=0}

import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# ============================================================
PATIENT_SIM_PATH = "patient_simulation_v2.py"  # adjust if needed
spec = importlib.util.spec_from_file_location("patient_simulation_v2", PATIENT_SIM_PATH)
patient_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patient_mod)
PatientModel = patient_mod.PatientModel
PATIENT_SPEED = 0.12 # m/s
PATIENT_SPEED_SD = 0.06 # m/s

matrix_spec = importlib.util.spec_from_file_location("make_ideal_distribution", "tests/make_ideal_distribution.py")
matrix_mod = importlib.util.module_from_spec(matrix_spec)
matrix_spec.loader.exec_module(matrix_mod)
true_p_hit = matrix_mod.estimate_true_phit_matrix(patient_seed=7, 
                                                  mc_per_cell=1000, 
                                                  patient_speed=PATIENT_SPEED, 
                                                  patient_speed_sd=PATIENT_SPEED_SD)
ideal_matrix = matrix_mod.make_ideal_distribution(true_p_hit, target_prob=0.6, variability=0.25, total_trials=200)
# ============================================================

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


class DifficultyIndexLogReg:
    """
    p_hat = sigmoid(beta0 + beta1 * I)
    I = w_d * d_norm + w_t * hard_time

    beta updated by supervised log-loss gradient (hit/miss).
    w updated by RL-style gradient ascent on shaped reward through p_hat.
    """
    def __init__(
        self,
        rng,
        beta0=0.0,
        beta1=1.0,
        w_d=1.0,
        w_t=1.0,
        lr_beta=0.25,
        lr_w=0.04,
        l2_w=1e-3,
        p_star=0.65,
        lam_p=0.55,
        lam_d=0.08,
    ):
        self.rng = rng
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)

        self.w_d = float(w_d)
        self.w_t = float(w_t)

        self.lr_beta = float(lr_beta)
        self.lr_w = float(lr_w)
        self.l2_w = float(l2_w)

        self.p_star = float(p_star)
        self.lam_p = float(lam_p)
        self.lam_d = float(lam_d)

        self._seen = set()  # track if we've seen both classes for stable behavior

    def index(self, d, t):
        dn = normalize_d(d)
        ht = hard_time_feature(t)
        return self.w_d * dn + self.w_t * ht, dn, ht

    def predict_p(self, d, t):
        # before seeing both classes, be maximally uncertain
        if len(self._seen) < 2:
            return 0.5
        I, _, _ = self.index(d, t)
        return float(sigmoid(self.beta0 + self.beta1 * I))

    def uncertainty(self, d, t):
        p = self.predict_p(d, t)
        return 1.0 - 2.0 * abs(p - 0.5)  # in [0,1]

    def reward(self, hit, d, t):
        # shaped reward encourages: (1) hits, (2) more distance, (3) p near p_star
        p_hat = self.predict_p(d, t)
        dn = normalize_d(d)
        r = (1.0 if hit else 0.0) + self.lam_d * dn - self.lam_p * ((p_hat - self.p_star) ** 2)
        return float(r)

    def update(self, hit, d_exec, t_exec, d_intended, t_intended):
        """
        Two updates:
        (A) Supervised update for beta from executed (d_exec, t_exec) and hit label.
        (B) RL-style update for w using reward computed on intended (d_intended, t_intended),
            with gradient flowing through p_hat(I(d,t)).
        """
        y = 1.0 if hit else 0.0
        self._seen.add(int(y))

        # ----- (A) logistic regression update on executed stimulus -----
        I_exec, _, _ = self.index(d_exec, t_exec)
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

        # If we haven't seen both classes yet, don't do RL update (too noisy/unstable)
        if len(self._seen) < 2:
            return float(p_exec), float(I_exec), None

        # ----- (B) RL-style update of difficulty weights w_d, w_t -----
        # Reward depends on p_hat at intended (d,t) (keeps policy near p_star)
        I_int, dn_int, ht_int = self.index(d_intended, t_intended)
        p_int = sigmoid(self.beta0 + self.beta1 * I_int)

        # r = hit + lam_d*dn - lam_p*(p - p_star)^2
        # dr/dp = -lam_p * 2*(p - p_star)
        dr_dp = -self.lam_p * 2.0 * (p_int - self.p_star)

        # dp/dI = p(1-p) * beta1
        dp_dI = p_int * (1.0 - p_int) * self.beta1

        # dI/dw_d = dn, dI/dw_t = ht
        # so dr/dw = dr/dp * dp/dI * dI/dw + small regularization
        grad_common = dr_dp * dp_dI

        grad_wd = grad_common * dn_int - self.l2_w * self.w_d
        grad_wt = grad_common * ht_int - self.l2_w * self.w_t

        self.w_d += self.lr_w * grad_wd
        self.w_t += self.lr_w * grad_wt

        # keep weights in a reasonable range
        self.w_d = float(np.clip(self.w_d, 0.05, 4.0))
        self.w_t = float(np.clip(self.w_t, 0.05, 4.0))

        return float(p_exec), float(I_exec), float(p_int)


# -----------------------------
# Controller run
# -----------------------------
def run_controller(
    n_trials=200,
    seed=7,
    p_star=0.65,
    explore_prob=0.40,  # active-learning exploration near p=0.5
    p_jitter=0.75,
):
    rng = np.random.default_rng(seed)
    patient = PatientModel(seed=seed)

    model = DifficultyIndexLogReg(
        rng=rng,
        beta0=0.0,
        beta1=1.0,
        w_d=1.0,
        w_t=1.0,
        lr_beta=0.30,
        lr_w=0.05,
        l2_w=2e-3,
        p_star=p_star,
        lam_p=0.55,
        lam_d=0.08,
    )

    counts = np.zeros((5, 5), dtype=int)

    hist = {
        "a_i": [], "a_j": [],
        "d": [], "t": [],
        "d_sys": [], "t_sys": [],
        "p_pred_intended": [],
        "p_pred_exec": [],
        "uncert": [],
        "hit": [],
        "reward": [],
        "rolling_hit": [],
        "beta0": [], "beta1": [],
        "w_d": [], "w_t": [],
        "index_exec": [],
    }

    prev_hit = True
    rolling_window = 20
    hit_list = []

    for k in range(n_trials):
        # compute predicted p and uncertainty for all 25 actions (intended grid points)
        p_hat = np.zeros((5, 5), dtype=float)
        uncert = np.zeros((5, 5), dtype=float)
        for i in range(5):
            for j in range(5):
                d, t = float(D_LEVELS[i]), float(T_LEVELS[j])
                p_hat[i, j] = model.predict_p(d, t)
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
        out = patient.sample_trial(t_sys=t_sys, d_sys=d_sys, distance_level=lvl, previous_hit=prev_hit)
        hit = bool(out["hit"])
        prev_hit = hit

        # reward computed on intended (d,t) for stable shaping
        r = model.reward(hit, d, t)

        # update model (beta on executed, w on intended via shaped reward)
        p_exec_before = model.predict_p(d_sys, t_sys)
        p_int_before = model.predict_p(d, t)
        p_exec_after, I_exec, p_int_after = model.update(hit, d_sys, t_sys, d, t)

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
        hist["p_pred_intended"].append(float(p_int_before))
        hist["p_pred_exec"].append(float(p_exec_before))
        hist["uncert"].append(float(uncert[i, j]))
        hist["hit"].append(int(hit))
        hist["reward"].append(float(r))
        hist["rolling_hit"].append(roll)
        hist["beta0"].append(float(model.beta0))
        hist["beta1"].append(float(model.beta1))
        hist["w_d"].append(float(model.w_d))
        hist["w_t"].append(float(model.w_t))
        hist["index_exec"].append(float(I_exec))

    return hist, counts


# -----------------------------
# Evaluation + plots
# -----------------------------
def normalize_counts(counts):
    s = counts.sum()
    if s <= 0:
        return np.ones_like(counts, dtype=float) / counts.size
    return counts / s


def kl_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


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


def visualize(hist, counts, phit_true, ideal_dist, p_star):
    d = np.array(hist["d"])
    t = np.array(hist["t"])
    d_sys = np.array(hist["d_sys"])
    t_sys = np.array(hist["t_sys"])
    p_int = np.array(hist["p_pred_intended"])
    p_exec = np.array(hist["p_pred_exec"])
    uncert = np.array(hist["uncert"])
    roll = np.array(hist["rolling_hit"])

    beta0 = np.array(hist["beta0"])
    beta1 = np.array(hist["beta1"])
    wd = np.array(hist["w_d"])
    wt = np.array(hist["w_t"])

    # 2) p prediction vs rolling hit
    plt.figure()
    plt.plot(p_int, label="predicted P(hit) @ intended")
    plt.plot(roll, linestyle="--", label="rolling hit rate")
    plt.axhline(p_star, linestyle=":", label=f"p_star={p_star}")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("trial")
    plt.title("Model prediction vs realized performance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) uncertainty
    plt.figure()
    plt.plot(uncert)
    plt.xlabel("trial")
    plt.ylabel("uncertainty (peaks at p=0.5)")
    plt.title("Active-learning uncertainty of chosen actions")
    plt.tight_layout()
    plt.show()

    # 4) parameters over time
    plt.figure()
    plt.plot(beta0, label="beta0")
    plt.plot(beta1, label="beta1")
    plt.xlabel("trial")
    plt.title("Logistic regression parameters")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(wd, label="w_d (distance weight)")
    plt.plot(wt, label="w_t (time weight)")
    plt.xlabel("trial")
    plt.title("Difficulty index weights (RL-updated)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # heatmaps
    xlabels = ["shortest", "short", "medium", "long", "longest"]  # time bins
    ylabels = ["closest", "close", "medium", "far", "farthest"]   # distance bins

    plot_heatmap(counts, "Selection counts (5x5 action grid)", xlabels, ylabels, annotate=True)
    plot_heatmap(phit_true, "True P(hit) per cell (Monte Carlo)", xlabels, ylabels, annotate=True)

    learned_dist = normalize_counts(counts)
    plot_heatmap(ideal_dist, "Ideal selection distribution (targeting p_star contour)", xlabels, ylabels, annotate=True)
    plot_heatmap(learned_dist, "Learned selection distribution", xlabels, ylabels, annotate=True)

    diff = learned_dist - ideal_dist
    plot_heatmap(diff, "Learned - Ideal distribution", xlabels, ylabels, annotate=True)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    P_STAR = 0.65

    # run controller (with jitter during execution)
    hist, counts = run_controller(
        n_trials=200,
        seed=7,
        p_star=P_STAR,
        explore_prob=0.15,
        p_jitter=0.25,
    )
    print(counts)

    plot_heatmap(ideal_matrix, "Ideal distribution",
                 xlabels=["shortest", "short", "medium", "long", "longest"],
                 ylabels=["closest", "close", "medium", "far", "farthest"], annotate=True)
    plot_heatmap(counts, "Actual selection counts",
                 xlabels=["shortest", "short", "medium", "long", "longest"],
                 ylabels=["closest", "close", "medium", "far", "farthest"], annotate=True)
    print(f"Absolute difference: {np.abs(counts - ideal_matrix).sum()}")