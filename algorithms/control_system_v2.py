"""
Continuous-margin control for selecting (d,t) pairs.

Key idea:
- Convert each trial into a continuous "margin" m_k:
    if HIT : m_k = 1 - time_ratio         (bigger = more comfortable)
    if MISS: m_k = dist_ratio - 1         (closer to 0 = nearly hit; very negative = far miss)
- Track EWMA of margin: m_hat
- PI controller pushes a scalar difficulty u in [0,1]
- Map u -> (d_sys, t_sys) via required speed + ROM ramp, with exploration + diversity

Requires: patient_simulation_v2.py (your simulator)
File reference: :contentReference[oaicite:0]{index=0}
"""

import math
import numpy as np
import importlib.util
from pathlib import Path

# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
def _load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v3.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v3", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel
# ============================================================

hit_margin = []
miss_margin = []

# ----------------------------
# Helpers: bins + rolling stats
# ----------------------------
def rolling_mean(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def level5(x, xmin, xmax):
    """Map x in [xmin,xmax] -> {0..4} by fifths."""
    u = (x - xmin) / (xmax - xmin + 1e-12)
    if u < 0.2: return 0
    if u < 0.4: return 1
    if u < 0.6: return 2
    if u < 0.8: return 3
    return 4


def bin25(d, t, dmin, dmax, tmin, tmax):
    return (level5(d, dmin, dmax), level5(t, tmin, tmax))


def rarity_bonus(counts_5x5, i, j, eps=1e-12):
    """-log(freq) with Laplace smoothing, higher for under-sampled bins."""
    total = counts_5x5.sum()
    freq = (counts_5x5[i, j] + 1.0) / (total + 25.0)
    return -math.log(freq + eps)


def distance_level_3(d, dmin, dmax):
    """Map d to distance_level in {0,1,2} (your PatientModel clips to 0..2 anyway)."""
    u = (d - dmin) / (dmax - dmin + 1e-12)
    if u < 1/3: return 0
    if u < 2/3: return 1
    return 2


# ----------------------------
# Continuous-margin PI controller
# ----------------------------
class MarginPIController:
    """
    Controls difficulty u in [0,1] using continuous margin feedback.

    Margin definition:
      HIT : m = 1 - time_ratio  (>=0; bigger => easier)
      MISS: m = dist_ratio - 1  (<=0; closer to 0 => near miss; very negative => too hard)

    We track EWMA m_hat, and use error e = (m_hat - m_star).
    - If m_hat > m_star: too easy => e>0 => increase difficulty u
    - If m_hat < m_star: too hard => e<0 => decrease difficulty u
    """
    def __init__(self, *, m_star=0.05, Kp=0.9, Ki=0.15, ewma_alpha=0.15, u0=0.35):
        self.m_star = float(m_star) #m_star: target margin
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.alpha = float(ewma_alpha)

        self.u = float(np.clip(u0, 0.0, 1.0))
        self.m_hat = 0.0
        self.e_int = 0.0

    def update(self, m_k):
        # EWMA of margin
        self.m_hat = (1.0 - self.alpha) * self.m_hat + self.alpha * float(m_k)

        # error: positive means too easy -> increase u (harder)
        e = self.m_hat - self.m_star

        # anti-windup: only integrate if not saturated in the direction of the error
        pushing_up = (e > 0 and self.u >= 1.0 - 1e-9)
        pushing_down = (e < 0 and self.u <= 0.0 + 1e-9)
        if not (pushing_up or pushing_down):
            self.e_int += e

        # PI update
        du = self.Kp * e + self.Ki * self.e_int
        self.u = float(np.clip(self.u + du, 0.0, 1.0))
        return self.u, self.m_hat, e


# ----------------------------
# Map difficulty u -> (d,t)
# u is actually the margin of the time and distance ratios
# ----------------------------
def propose_dt_from_u(
    u,
    *,
    dmin=0.10, dmax=0.80,
    tmin=1.0,  tmax=7.0,
    vmin=0.06, vmax=0.20,
    rng=None
):
    """
    Difficulty should increase with:
      - larger distance (ROM demand)
      - higher required speed v_req = d/t

    We do:
      d_base = dmin + u*(dmax-dmin)
      v_req  = vmin + u*(vmax-vmin)
      t_base = d_base / v_req
      then clamp t_base to [tmin,tmax]

    Note: once t is clamped, the realized v_req may deviate; still OK.
    """
    rng = np.random.default_rng() if rng is None else rng
    random_split = rng.uniform(0.4, 0.6) # decide how much of added difficulty comes from d vs t

    d_base = dmin + u * (dmax - dmin) * random_split
    v_req = vmin + u * (vmax - vmin) * (1.0 - random_split)

    t_base = d_base / max(v_req, 1e-9)
    t_base = float(np.clip(t_base, tmin, tmax))

    return float(d_base), float(t_base)


def choose_with_exploration_and_diversity(
    d_base, t_base,
    *,
    counts_5x5,
    dmin, dmax, tmin, tmax,
    rng,
    n_candidates=25,
    d_sigma=0.05,
    t_sigma=0.35,
    w_close=1.0,
    w_rare=0.35
):
    """
    Build a small local candidate set around (d_base,t_base), then pick one by
    a soft score:
      score = -w_close * normalized_distance_to_base + w_rare * rarity_bonus(bin)

    This yields variety over the 5x5 grid while staying near the controller's intent.
    """
    cand = []
    scores = []

    for _ in range(n_candidates):
        d = float(np.clip(rng.normal(d_base, d_sigma), dmin, dmax))
        t = float(np.clip(rng.normal(t_base, t_sigma), tmin, tmax))

        # closeness penalty in normalized space
        dn = (d - d_base) / (dmax - dmin + 1e-12)
        tn = (t - t_base) / (tmax - tmin + 1e-12)
        close_pen = dn*dn + tn*tn

        # rarity bonus from 5x5 bins
        i, j = bin25(d, t, dmin, dmax, tmin, tmax)
        rare = rarity_bonus(counts_5x5, i, j)

        score = -w_close * close_pen + w_rare * rare
        cand.append((d, t, i, j))
        scores.append(score)

    # softmax sampling (keeps exploration stochastic but biased)
    s = np.array(scores, dtype=float)
    s = s - s.max()
    p = np.exp(s)
    p = p / p.sum()

    idx = int(rng.choice(len(cand), p=p))
    d, t, i, j = cand[idx]
    return d, t, i, j


def sample_diagonal_pair(
    *,
    dmin,
    dmax,
    tmin,
    tmax,
    d_step,
    t_step,
    rng=None
):
    """
    Sample a (d,t) pair along the diagonal:
    larger d <-> larger t, quantized to the discretized grid.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.uniform(0.0, 1.0)
    d = dmin + u * (dmax - dmin)
    t = tmin + u * (tmax - tmin)

    # quantize to grid
    d = round(d / d_step) * d_step
    t = round(t / t_step) * t_step

    return float(d), float(t)


# ----------------------------
# Main simulation loop
# ----------------------------
def run_sim(
    patient: PatientModel,
    n_trials=100,
    seed=7,
    # controller targets/tuning
    m_star=0.05,     # aim for "slightly comfortable but near the edge"
    Kp=0.9,
    Ki=0.15,
    ewma_alpha=0.15,
    # action space bounds
    dmin=0.10, dmax=0.80,
    tmin=1.0,  tmax=7.0,
    d_step=0.05,
    t_step=0.1,
    # mapping bounds for required speed
    vmin=0.06, vmax=0.20,
    # diagonal resample control
    diag_after=10,
):
    rng = np.random.default_rng(seed)
    ctrl = MarginPIController(m_star=m_star, Kp=Kp, Ki=Ki, ewma_alpha=ewma_alpha, u0=0.35)

    counts_5x5 = np.zeros((5, 5), dtype=int)
    previous_hit = True

    hist = {
        "u": [],
        "d": [],
        "t": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "margin": [],
        "m_hat": [],
        "err": [],
        "bin_i": [],
        "bin_j": [],
    }

    for k in range(n_trials):
        if diag_after is not None and (k % diag_after == 0):
            d_sys, t_sys = sample_diagonal_pair(
                dmin=dmin, dmax=dmax, tmin=tmin, tmax=tmax,
                d_step=d_step, t_step=t_step,
                rng=rng
            )
            # print("Trial", k, ": diagonal resample to (d,t)=(", d_sys, ",", t_sys, ")")
            i, j = bin25(d_sys, t_sys, dmin, dmax, tmin, tmax)
        else:
            # controller proposes base (d,t)
            d_base, t_base = propose_dt_from_u(
                ctrl.u,
                dmin=dmin, dmax=dmax, tmin=tmin, tmax=tmax,
                vmin=vmin, vmax=vmax,
                rng=rng
            )
            # print("Trial", k, ": controller proposes (d,t)=(", round(d_base,3), ",", round(t_base,3), ")")

            # pick a nearby candidate, biased toward under-sampled bins
            d_sys, t_sys, i, j = choose_with_exploration_and_diversity(
                d_base, t_base,
                counts_5x5=counts_5x5,
                dmin=dmin, dmax=dmax, tmin=tmin, tmax=tmax,
                rng=rng
            )
            # print("Trial", k, ": (d,t)=(", round(d_sys,3), ",", round(t_sys,3), ")")

        # update diversity grid
        counts_5x5[i, j] += 1

        # distance_level for the simulator
        lvl = distance_level_3(d_sys, dmin, dmax)

        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit
        )

        hit = bool(outcome["hit"])
        previous_hit = hit

        time_ratio = float(outcome["time_ratio"])
        dist_ratio = float(outcome["dist_ratio"])

        # continuous margin
        if hit:
            m_k = 1.0 - time_ratio          # >=0 ; bigger => easier
            hit_margin.append(m_k)
        else:
            m_k = (dist_ratio - 1.0)*0.7    # <=0 ; closer to 0 => near miss
            miss_margin.append(m_k)
        # controller update
        u_next, m_hat, err = ctrl.update(m_k)

        if diag_after is not None and (k % diag_after == 0):
            ctrl.u = (d_sys - dmin) / (dmax - dmin + 1e-12)  # reset u to match diagonal choice
            # print("  resetting controller u to", round(ctrl.u,3))

        # log
        hist["u"].append(ctrl.u)
        hist["d"].append(d_sys)
        hist["t"].append(t_sys)
        hist["hit"].append(int(hit))
        hist["time_ratio"].append(time_ratio)
        hist["dist_ratio"].append(dist_ratio)
        hist["margin"].append(m_k)
        hist["m_hat"].append(m_hat)
        hist["err"].append(err)
        hist["bin_i"].append(i)
        hist["bin_j"].append(j)

    return hist, counts_5x5


