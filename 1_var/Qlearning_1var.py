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
N_DIRECTIONS = 5


T_FIXED_DEFAULT = 5.0


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
    d_min: float = 0.10
    d_max: float = 0.80
    n_d_bins: int = 5
    alpha: float = 0.20
    gamma: float = 0.92
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    p_star: float = 0.60
    rolling_window: int = 20

    p_bins: int = 5
    ratio_bins: int = 5
    streak_cap: int = 5

    t_fixed: float = T_FIXED_DEFAULT


def level_index(x: float, xmin: float, xmax: float, bins: int) -> int:
    if bins <= 1 or xmax <= xmin:
        return 0
    r = (x - xmin) / (xmax - xmin)
    r = clamp(r, 0.0, 1.0)
    idx = int(np.floor(r * bins + 1e-12))
    return int(min(bins - 1, idx))


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
    abs_d_min, abs_d_max = 0.05, 1.5

    defaults = QLearningConfig()
    if not calibration_result:
        return defaults.d_min, defaults.d_max

    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = abs_d_min

    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15

    return float(d_min_cal), float(min(d_max_cal, abs_d_max))


def cap_distance_bounds(patient: PatientModel, d_min: float, d_max: float):
    patient_reach = float(getattr(patient, "max_reach", d_max))
    if not np.isfinite(patient_reach) or patient_reach <= 0:
        patient_reach = float(d_max)

    capped_min = max(float(d_min), 0.0)
    capped_max = float(min(d_max, patient_reach))
    if capped_max < capped_min:
        capped_max = capped_min
    return capped_min, capped_max


def cfg_defaults() -> QLearningConfig:
    return QLearningConfig()


class QLearner:
    def __init__(self, cfg: QLearningConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.d_levels = np.linspace(cfg.d_min, cfg.d_max, cfg.n_d_bins)
        self.actions = [d_bin for d_bin in range(cfg.n_d_bins)]
        self.n_actions = len(self.actions)
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.sa_hits = np.zeros(self.n_actions, dtype=float)
        self.sa_counts = np.zeros(self.n_actions, dtype=float)
        self.epsilon = cfg.epsilon_start

    def action_to_values(self, action_idx: int) -> Tuple[int, float, float]:
        d_bin = self.actions[action_idx]
        d_sys = float(self.d_levels[d_bin])
        t_sys = float(self.cfg.t_fixed)
        return d_bin, d_sys, t_sys

    def _q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=float)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...]) -> int:
        q_vals = self._q_values(state)
        if float(self.rng.uniform()) < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(q_vals))

    def phit_sa(self, action_idx: int, fallback_hit: int) -> float:
        if self.sa_counts[action_idx] <= 0:
            return float(fallback_hit)
        return float(self.sa_hits[action_idx] / self.sa_counts[action_idx])

    def update_sa_stats(self, action_idx: int, hit: int) -> None:
        self.sa_counts[action_idx] += 1.0
        self.sa_hits[action_idx] += float(hit)

    def reward(self, action_idx: int, fallback_hit: int) -> float:
        p_hit = self.phit_sa(action_idx, fallback_hit=fallback_hit)
        return float(1.0 - (p_hit - self.cfg.p_star) ** 2)

    def update_q(
        self,
        state: Tuple[int, ...],
        action_idx: int,
        reward: float,
        next_state: Tuple[int, ...],
    ) -> None:
        q_vals = self._q_values(state)
        next_q = self._q_values(next_state)
        td_target = reward + self.cfg.gamma * float(np.max(next_q))
        td_error = td_target - q_vals[action_idx]
        q_vals[action_idx] += self.cfg.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)


def build_state(
    cfg: QLearningConfig,
    d_bin: int,
    mean_p_hit: float,
    mean_dist_ratio: float,
    hit_streak: int,
) -> Tuple[int, ...]:
    p_bin = level_index(mean_p_hit, 0.0, 1.0, cfg.p_bins)
    d_ratio_bin = level_index(mean_dist_ratio, 0.0, 1.0, cfg.ratio_bins)
    capped_streak = int(np.clip(hit_streak, -cfg.streak_cap, cfg.streak_cap))
    streak_idx = capped_streak + cfg.streak_cap
    return (d_bin, p_bin, d_ratio_bin, streak_idx)


def run_sim(
    patient: Optional[PatientModel] = None,
    n_trials: int = 2000,
    seed: int = 7,
    cfg: Optional[QLearningConfig] = None,
    calibration: bool = True,
    t_fixed: Optional[float] = None,
):
    cfg = cfg or QLearningConfig()
    if t_fixed is not None:
        cfg.t_fixed = float(t_fixed)
    rng = np.random.default_rng(seed)
    patient = patient or PatientModel(seed=seed)

    calibration_result = None
    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        dmin, dmax = derive_bounds_from_calibration(calibration_result, patient)
        cfg.d_min = dmin
        cfg.d_max = dmax
    cfg.d_min, cfg.d_max = cap_distance_bounds(patient, cfg.d_min, cfg.d_max)

    agent = QLearner(cfg=cfg, rng=rng)

    counts = np.zeros((cfg.n_d_bins, 5), dtype=int)

    logs: Dict[str, List] = {
        "trial": [],
        "action_idx": [],
        "d_bin": [],
        "d": [],
        "t": [],
        "hit": [],
        "p_hit_sa": [],
        "reward": [],
        "epsilon": [],
        "time_ratio": [],
        "dist_ratio": [],
        "mean_p_hit": [],
        "mean_dist_ratio": [],
        "hit_streak": [],
        "q_max": [],
    }

    recent_hits = deque(maxlen=cfg.rolling_window)
    recent_dist_ratio = deque(maxlen=cfg.rolling_window)

    previous_hit = True
    hit_streak = 0

    last_d_bin = 2

    for k in range(n_trials):
        mean_p_hit = float(np.mean(recent_hits)) if recent_hits else 0.5
        mean_dist_ratio = float(np.mean(recent_dist_ratio)) if recent_dist_ratio else 0.5

        state = build_state(
            cfg=cfg,
            d_bin=last_d_bin,
            mean_p_hit=mean_p_hit,
            mean_dist_ratio=mean_dist_ratio,
            hit_streak=hit_streak,
        )
        action_idx = agent.select_action(state)
        d_bin, d_sys, t_sys = agent.action_to_values(action_idx)

        lvl = distance_level_from_patient_bins(patient, d_sys)
        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
            direction_bin=int(rng.integers(0, 5)),
        )

        hit = int(bool(outcome["hit"]))
        previous_hit = bool(hit)
        if hit:
            hit_streak = max(1, hit_streak + 1) if hit_streak >= 0 else 1
        else:
            hit_streak = min(-1, hit_streak - 1) if hit_streak <= 0 else -1

        recent_hits.append(hit)
        recent_dist_ratio.append(float(outcome["dist_ratio"]))

        agent.update_sa_stats(action_idx, hit)
        p_hit_sa = agent.phit_sa(action_idx, fallback_hit=hit)
        reward = agent.reward(action_idx, fallback_hit=hit)

        next_mean_p_hit = float(np.mean(recent_hits))
        next_mean_dist_ratio = float(np.mean(recent_dist_ratio))

        next_state = build_state(
            cfg=cfg,
            d_bin=d_bin,
            mean_p_hit=next_mean_p_hit,
            mean_dist_ratio=next_mean_dist_ratio,
            hit_streak=hit_streak,
        )

        agent.update_q(state, action_idx, reward, next_state)
        agent.decay_epsilon()

        counts[d_bin, 0] += 1

        logs["trial"].append(k)
        logs["action_idx"].append(int(action_idx))
        logs["d_bin"].append(int(d_bin))
        logs["d"].append(float(d_sys))
        logs["t"].append(float(t_sys))
        logs["hit"].append(int(hit))
        logs["p_hit_sa"].append(float(p_hit_sa))
        logs["reward"].append(float(reward))
        logs["epsilon"].append(float(agent.epsilon))
        logs["time_ratio"].append(float(outcome["time_ratio"]))
        logs["dist_ratio"].append(float(outcome["dist_ratio"]))
        logs["mean_p_hit"].append(float(next_mean_p_hit))
        logs["mean_dist_ratio"].append(float(next_mean_dist_ratio))
        logs["hit_streak"].append(int(hit_streak))
        logs["q_max"].append(float(np.max(agent._q_values(next_state))))

        last_d_bin = d_bin

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
    ax.plot(trials, qmax, label="max Q(next_state)")
    ax.set_title("Q-value Growth")
    ax.set_xlabel("Trial")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    xlabels = ["shortest", "short", "medium", "long", "longest"]
    ylabels = ["closest", "close", "medium", "far", "farthest"]

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(counts, aspect="auto")
    ax2.set_title("Selection Counts (distance-only)")
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
