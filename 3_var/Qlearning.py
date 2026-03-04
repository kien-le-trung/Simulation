from __future__ import annotations

import importlib.util
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATIENT_SIM_PATH = BASE_DIR / "patients" / "patient_simulation_v4.py"
patient_mod = _load_module_from_path("patients.patient_simulation_v4", PATIENT_SIM_PATH)
PatientModel = patient_mod.PatientModel

N_DIRECTIONS = 5
D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class QLearningConfig:
    d_min: float = D_MIN
    d_max: float = D_MAX
    t_min: float = T_MIN
    t_max: float = T_MAX
    n_d_bins: int = 5
    n_t_bins: int = 5
    n_dirs: int = N_DIRECTIONS

    alpha: float = 0.02           # step size for linear FA (smaller than tabular)
    gamma: float = 0.0            # bandit-style: only immediate reward matters
    epsilon_start: float = 1.0    # start fully exploratory
    epsilon_end: float = 0.10     # keep meaningful exploration
    epsilon_decay: float = 0.990  # ~230 trials to reach floor

    p_star: float = 0.70
    rolling_window: int = 30


def distance_level_from_patient_bins(patient: PatientModel, d_sys: float) -> int:
    d_means = np.asarray(patient.d_means, dtype=float)
    idx = np.where(d_means <= d_sys)[0]
    return int(idx[-1]) if len(idx) else 0


def apply_calibration_priors(patient: PatientModel, calibration_result: dict | None):
    if not calibration_result:
        return
    per_direction = calibration_result.get("per_direction", {})
    for direction, stats in per_direction.items():
        idx = int(np.clip(direction, 0, N_DIRECTIONS - 1))
        n_reached = float(stats.get("n_reached", 0))
        n_censored = float(stats.get("n_censored", 0))
        patient.spatial_success_alpha[idx] += n_reached
        patient.spatial_success_beta[idx] += n_censored


def derive_bounds_from_calibration(calibration_result, patient):
    ABS_D_MIN, ABS_D_MAX = 0.05, 1.5
    ABS_T_MIN, ABS_T_MAX = 0.15, 15.0

    if not calibration_result:
        return D_MIN, D_MAX, T_MIN, T_MAX

    trials = calibration_result.get("trials", [])
    speeds = []
    for tr in trials:
        hit = tr.get("hit", tr.get("reached", False))
        t_pat = float(tr.get("t_pat_obs", tr.get("t_pat", 0)))
        d_sys = float(tr.get("d_sys", 0))
        if hit and t_pat > 0.01 and d_sys > 0.01:
            speeds.append(d_sys / t_pat)

    if len(speeds) < 3:
        return D_MIN, D_MAX, T_MIN, T_MAX

    speeds = np.array(speeds)
    v_slow = max(float(np.percentile(speeds, 10)), 0.01)
    v_fast = max(float(np.percentile(speeds, 90)), v_slow * 1.5)

    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = ABS_D_MIN

    t_min_cal = max(ABS_T_MIN, d_min_cal / (v_fast * 2.0))
    t_max_cal = min(ABS_T_MAX, (d_max_cal / v_slow) * 2.0)

    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15
    if t_max_cal - t_min_cal < 1.0:
        t_max_cal = t_min_cal + 1.0

    return (float(d_min_cal), float(min(d_max_cal, ABS_D_MAX)),
            float(t_min_cal), float(t_max_cal))


# ---------------------------------------------------------------------------
# Linear function approximation Q-learner
# ---------------------------------------------------------------------------
# Q(s, a) = w · phi(s, a)
#
# State: (mean_p_hit, streak)  — compact continuous summary
# Action: (d, t, direction)    — discrete grid
#
# Features phi(s, a):
#   1. bias
#   2. d_norm                  — distance (normalized)
#   3. t_norm                  — time (normalized)
#   4. v_req_norm              — required speed d/t (captures d-t interaction)
#   5. dir_centered            — direction (centered: -1 to +1)
#   6. gap * d_norm            — when above target, prefer harder d
#   7. gap * t_norm            — when above target, prefer longer t
#   8. gap * v_req_norm        — speed adaptation by performance gap
#   9. gap                     — bias shift by performance
#  10. gap^2                   — non-linear penalty for being far from target
#
# where gap = mean_p - p_star (positive = too easy, negative = too hard)
# ---------------------------------------------------------------------------

N_FEATURES = 10


class LinearQLearner:
    def __init__(self, cfg: QLearningConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.w = np.zeros(N_FEATURES, dtype=float)
        self.epsilon = cfg.epsilon_start

        # Pre-build discrete action grid
        self.d_levels = np.linspace(cfg.d_min, cfg.d_max, cfg.n_d_bins)
        self.t_levels = np.linspace(cfg.t_min, cfg.t_max, cfg.n_t_bins)
        self.actions = [
            (d_idx, t_idx, dir_idx)
            for d_idx in range(cfg.n_d_bins)
            for t_idx in range(cfg.n_t_bins)
            for dir_idx in range(cfg.n_dirs)
        ]
        self.n_actions = len(self.actions)

    def _features(self, mean_p: float, d: float, t: float, direction: int) -> np.ndarray:
        d_norm = (d - self.cfg.d_min) / max(self.cfg.d_max - self.cfg.d_min, 1e-6)
        t_norm = (t - self.cfg.t_min) / max(self.cfg.t_max - self.cfg.t_min, 1e-6)
        v_req = d / max(t, 0.01)
        # Normalize v_req roughly to [0, 1]
        v_max = self.cfg.d_max / max(self.cfg.t_min, 0.01)
        v_req_norm = clamp(v_req / max(v_max, 1e-6), 0.0, 1.0)
        dir_centered = (float(direction) - 2.0) / 2.0  # [-1, 1]
        gap = mean_p - self.cfg.p_star  # positive = too easy

        return np.array([
            1.0,                    # bias
            d_norm,                 # distance
            t_norm,                 # time
            v_req_norm,             # required speed
            dir_centered,           # direction
            gap * d_norm,           # adapt distance by performance gap
            gap * t_norm,           # adapt time by performance gap
            gap * v_req_norm,       # adapt speed by performance gap
            gap,                    # performance gap bias
            gap * gap,              # non-linear gap penalty
        ], dtype=float)

    def q_value(self, mean_p: float, d: float, t: float, direction: int) -> float:
        phi = self._features(mean_p, d, t, direction)
        return float(np.dot(self.w, phi))

    def action_to_values(self, action_idx: int) -> Tuple[int, int, int, float, float]:
        d_idx, t_idx, dir_idx = self.actions[action_idx]
        d_sys = float(self.d_levels[d_idx])
        t_sys = float(self.t_levels[t_idx])
        return d_idx, t_idx, dir_idx, d_sys, t_sys

    def select_action(self, mean_p: float) -> int:
        if float(self.rng.uniform()) < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        # Greedy: pick action with highest Q-value
        best_idx = 0
        best_q = -1e18
        for i, (d_idx, t_idx, dir_idx) in enumerate(self.actions):
            d = float(self.d_levels[d_idx])
            t = float(self.t_levels[t_idx])
            q = self.q_value(mean_p, d, t, dir_idx)
            if q > best_q:
                best_q = q
                best_idx = i
        return best_idx

    def update(self, mean_p: float, action_idx: int, reward: float,
               next_mean_p: float) -> None:
        d_idx, t_idx, dir_idx = self.actions[action_idx]
        d = float(self.d_levels[d_idx])
        t = float(self.t_levels[t_idx])

        phi = self._features(mean_p, d, t, dir_idx)
        q_current = float(np.dot(self.w, phi))

        # With gamma=0, td_target = reward (bandit-style)
        if self.cfg.gamma > 0:
            # Find max Q over next actions
            best_next_q = -1e18
            for i, (nd, nt, ndir) in enumerate(self.actions):
                nd_val = float(self.d_levels[nd])
                nt_val = float(self.t_levels[nt])
                nq = self.q_value(next_mean_p, nd_val, nt_val, ndir)
                if nq > best_next_q:
                    best_next_q = nq
            td_target = reward + self.cfg.gamma * best_next_q
        else:
            td_target = reward

        td_error = td_target - q_current
        self.w += self.cfg.alpha * td_error * phi

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_sim(
    patient: Optional[PatientModel] = None,
    n_trials: int = 500,
    seed: int = 7,
    cfg: Optional[QLearningConfig] = None,
    calibration: bool = True,
):
    cfg = cfg or QLearningConfig()
    rng = np.random.default_rng(seed)
    patient = patient or PatientModel(seed=seed)

    # Calibration and bounds derivation
    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        dmin, dmax, tmin, tmax = derive_bounds_from_calibration(calibration_result, patient)
        cfg.d_min = dmin
        cfg.d_max = dmax
        cfg.t_min = tmin
        cfg.t_max = tmax

    agent = LinearQLearner(cfg=cfg, rng=rng)
    counts = np.zeros((cfg.n_d_bins, cfg.n_t_bins), dtype=int)

    logs: Dict[str, List] = {
        "d": [],
        "t": [],
        "direction": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "t_pat": [],
        "d_pat": [],
    }

    recent_hits: deque = deque(maxlen=cfg.rolling_window)
    previous_hit = True

    for k in range(n_trials):
        mean_p = float(np.mean(recent_hits)) if recent_hits else 0.5

        action_idx = agent.select_action(mean_p)
        d_bin, t_bin, direction, d_sys, t_sys = agent.action_to_values(action_idx)

        lvl = distance_level_from_patient_bins(patient, d_sys)
        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
            direction_bin=direction,
        )

        hit = int(bool(outcome["hit"]))
        previous_hit = bool(hit)
        if hit:
            patient.spatial_success_alpha[direction] += 1.0
        else:
            patient.spatial_success_beta[direction] += 1.0

        recent_hits.append(hit)
        new_mean_p = float(np.mean(recent_hits))

        # Reward: negative squared distance from target hit rate
        reward = -((new_mean_p - cfg.p_star) ** 2)

        agent.update(mean_p, action_idx, reward, new_mean_p)
        agent.decay_epsilon()

        counts[d_bin, t_bin] += 1

        logs["d"].append(float(d_sys))
        logs["t"].append(float(t_sys))
        logs["direction"].append(int(direction))
        logs["hit"].append(int(hit))
        logs["time_ratio"].append(float(outcome["time_ratio"]))
        logs["dist_ratio"].append(float(outcome["dist_ratio"]))
        logs["t_pat"].append(float(outcome["t_pat"]))
        logs["d_pat"].append(float(outcome["d_pat"]))

    return logs, counts, patient
