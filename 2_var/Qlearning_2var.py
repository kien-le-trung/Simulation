from __future__ import annotations

import importlib.util
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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

D_MIN, D_MAX = 0.10, 0.80
T_MIN, T_MAX = 1.0, 7.0


def _load_highspeed_lowrom_profile() -> Dict[str, float]:
    panel_path = BASE_DIR / "simulation_control_panel_v2.py"
    panel_mod = _load_module_from_path("simulation_control_panel_v2", panel_path)
    base_profiles = getattr(panel_mod, "BASE_PATIENT_PROFILES", {})
    if "highspeed_lowrom" not in base_profiles:
        raise KeyError("Profile 'highspeed_lowrom' not found in BASE_PATIENT_PROFILES.")
    return dict(base_profiles["highspeed_lowrom"])


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

    alpha: float = 0.02
    gamma: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay: float = 0.990

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
        idx = int(np.clip(direction, 0, 4))
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


N_FEATURES = 9


class LinearQLearner:
    def __init__(self, cfg: QLearningConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.w = np.zeros(N_FEATURES, dtype=float)
        self.epsilon = cfg.epsilon_start

        self.d_levels = np.linspace(cfg.d_min, cfg.d_max, cfg.n_d_bins)
        self.t_levels = np.linspace(cfg.t_min, cfg.t_max, cfg.n_t_bins)
        self.actions = [
            (d_idx, t_idx)
            for d_idx in range(cfg.n_d_bins)
            for t_idx in range(cfg.n_t_bins)
        ]
        self.n_actions = len(self.actions)

    def _features(self, mean_p: float, d: float, t: float) -> np.ndarray:
        d_norm = (d - self.cfg.d_min) / max(self.cfg.d_max - self.cfg.d_min, 1e-6)
        t_norm = (t - self.cfg.t_min) / max(self.cfg.t_max - self.cfg.t_min, 1e-6)
        v_req = d / max(t, 0.01)
        v_max = self.cfg.d_max / max(self.cfg.t_min, 0.01)
        v_req_norm = clamp(v_req / max(v_max, 1e-6), 0.0, 1.0)
        gap = mean_p - self.cfg.p_star

        return np.array([
            1.0,
            d_norm,
            t_norm,
            v_req_norm,
            gap * d_norm,
            gap * t_norm,
            gap * v_req_norm,
            gap,
            gap * gap,
        ], dtype=float)

    def q_value(self, mean_p: float, d: float, t: float) -> float:
        phi = self._features(mean_p, d, t)
        return float(np.dot(self.w, phi))

    def action_to_values(self, action_idx: int) -> Tuple[int, int, float, float]:
        d_idx, t_idx = self.actions[action_idx]
        d_sys = float(self.d_levels[d_idx])
        t_sys = float(self.t_levels[t_idx])
        return d_idx, t_idx, d_sys, t_sys

    def max_q(self, mean_p: float) -> float:
        best_q = -1e18
        for d_idx, t_idx in self.actions:
            d = float(self.d_levels[d_idx])
            t = float(self.t_levels[t_idx])
            q = self.q_value(mean_p, d, t)
            if q > best_q:
                best_q = q
        return float(best_q)

    def select_action(self, mean_p: float) -> int:
        if float(self.rng.uniform()) < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        best_idx = 0
        best_q = -1e18
        for i, (d_idx, t_idx) in enumerate(self.actions):
            d = float(self.d_levels[d_idx])
            t = float(self.t_levels[t_idx])
            q = self.q_value(mean_p, d, t)
            if q > best_q:
                best_q = q
                best_idx = i
        return best_idx

    def update(self, mean_p: float, action_idx: int, reward: float, next_mean_p: float) -> None:
        d_idx, t_idx = self.actions[action_idx]
        d = float(self.d_levels[d_idx])
        t = float(self.t_levels[t_idx])

        phi = self._features(mean_p, d, t)
        q_current = float(np.dot(self.w, phi))

        if self.cfg.gamma > 0:
            td_target = reward + self.cfg.gamma * self.max_q(next_mean_p)
        else:
            td_target = reward

        td_error = td_target - q_current
        self.w += self.cfg.alpha * td_error * phi

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)


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
        "trial": [],
        "d": [],
        "t": [],
        "direction": [],
        "hit": [],
        "time_ratio": [],
        "dist_ratio": [],
        "t_pat": [],
        "d_pat": [],
        "mean_p_hit": [],
        "reward": [],
        "epsilon": [],
        "q_max": [],
    }

    recent_hits: deque = deque(maxlen=cfg.rolling_window)
    previous_hit = True

    for k in range(n_trials):
        mean_p = float(np.mean(recent_hits)) if recent_hits else 0.5

        action_idx = agent.select_action(mean_p)
        d_bin, t_bin, d_sys, t_sys = agent.action_to_values(action_idx)
        direction = int(rng.integers(0, 5))

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

        recent_hits.append(hit)
        new_mean_p = float(np.mean(recent_hits))

        reward = -((new_mean_p - cfg.p_star) ** 2)

        agent.update(mean_p, action_idx, reward, new_mean_p)
        agent.decay_epsilon()

        counts[d_bin, t_bin] += 1

        logs["trial"].append(k)
        logs["d"].append(float(d_sys))
        logs["t"].append(float(t_sys))
        logs["direction"].append(direction)
        logs["hit"].append(int(hit))
        logs["time_ratio"].append(float(outcome["time_ratio"]))
        logs["dist_ratio"].append(float(outcome["dist_ratio"]))
        logs["t_pat"].append(float(outcome["t_pat"]))
        logs["d_pat"].append(float(outcome["d_pat"]))
        logs["mean_p_hit"].append(float(new_mean_p))
        logs["reward"].append(float(reward))
        logs["epsilon"].append(float(agent.epsilon))
        logs["q_max"].append(float(agent.max_q(new_mean_p)))

    return logs, counts, patient


def rolling_mean(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    out = np.empty_like(arr, dtype=float)
    csum = np.cumsum(arr, dtype=float)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def plot_qlearning_results(
    logs: Dict[str, List],
    counts: np.ndarray,
    cfg: QLearningConfig,
    window: int = 30,
) -> None:
    trials = np.asarray(logs.get("trial", []), dtype=int)
    hits = np.asarray(logs.get("hit", []), dtype=float)
    rewards = np.asarray(logs.get("reward", []), dtype=float)
    epsilon = np.asarray(logs.get("epsilon", []), dtype=float)
    qmax = np.asarray(logs.get("q_max", []), dtype=float)
    mean_phit = np.asarray(logs.get("mean_p_hit", []), dtype=float)

    rolling_hit = rolling_mean(hits.tolist(), window=window)
    rolling_reward = rolling_mean(rewards.tolist(), window=window)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(trials, rolling_hit, label=f"rolling hit ({window})")
    ax.plot(trials, mean_phit, linestyle="--", label="mean p_hit")
    ax.axhline(cfg.p_star, linestyle=":", label=f"p*={cfg.p_star:.2f}")
    for vx in (25, 50, 100):
        ax.axvline(x=vx, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Hit Behavior")
    ax.set_xlabel("Trial")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(trials, rolling_reward, label=f"rolling reward ({window})")
    ax.set_title("Reward")
    ax.set_xlabel("Trial")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(trials, epsilon, label="epsilon")
    ax.set_title("Exploration Schedule")
    ax.set_xlabel("Trial")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(trials, qmax, label="max Q")
    ax.set_title("Q-value Growth")
    ax.set_xlabel("Trial")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    xlabels = ["shortest", "short", "medium", "long", "longest"]
    ylabels = ["closest", "close", "medium", "far", "farthest"]

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(counts, aspect="auto")
    ax2.set_title("5x5 Selection Counts")
    ax2.set_xticks(range(len(xlabels)))
    ax2.set_xticklabels(xlabels)
    ax2.set_yticks(range(len(ylabels)))
    ax2.set_yticklabels(ylabels)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax2.text(j, i, str(int(counts[i, j])), ha="center", va="center", fontsize=8)
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()

    plt.show()


def run_600_trial_simulation(seed: int = 7, patient: Optional[PatientModel] = None):
    if patient is None:
        profile_params = _load_highspeed_lowrom_profile()
        profile_params["seed"] = seed
        patient = PatientModel(**profile_params)
    cfg = QLearningConfig()
    logs, counts, patient = run_sim(patient=patient, n_trials=5000, seed=seed, cfg=cfg)
    plot_qlearning_results(logs, counts, cfg=cfg, window=30)
    return logs, counts, patient


if __name__ == "__main__":
    run_600_trial_simulation(seed=7)
