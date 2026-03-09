from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


@dataclass
class QLearningConfig:
    d_min: float = 0.10
    d_max: float = 0.80
    n_states: int = 10          # number of grid points; state space = action space size
    alpha: float = 0.20         # learning rate
    gamma: float = 0.92         # discount factor
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    t_fixed: float = T_FIXED_DEFAULT


# ----------------------------
# Helpers
# ----------------------------
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
        patient.spatial_success_alpha[idx] += float(stats.get("n_reached", 0))
        patient.spatial_success_beta[idx] += float(stats.get("n_censored", 0))


def derive_bounds_from_calibration(calibration_result, patient):
    defaults = QLearningConfig()
    if not calibration_result:
        return defaults.d_min, defaults.d_max
    d_max_cal = max(float(patient.max_reach), 0.20)
    d_min_cal = 0.05
    if d_max_cal - d_min_cal < 0.15:
        d_max_cal = d_min_cal + 0.15
    return float(d_min_cal), float(min(d_max_cal, 1.5))


def cap_distance_bounds(patient: PatientModel, d_min: float, d_max: float):
    patient_reach = float(getattr(patient, "max_reach", d_max))
    if not np.isfinite(patient_reach) or patient_reach <= 0:
        patient_reach = float(d_max)
    capped_min = max(float(d_min), 0.0)
    capped_max = float(min(d_max, patient_reach))
    if capped_max < capped_min:
        capped_max = capped_min
    return capped_min, capped_max


# ----------------------------
# Reward
# ----------------------------
def compute_reward(hit: bool, dist_ratio: float, time_ratio: float) -> float:
    """
    Reward incorporating hit outcome, dist_ratio, and time_ratio.

    HIT:  r = time_ratio in [0, 1]
        time_ratio = t_pat / t_sys; close to 1 means the patient barely made it
        (challenging hit) → higher reward. Easy hits (low time_ratio) get less reward,
        nudging the agent toward harder but still achievable targets.

    MISS: r = dist_ratio - 1 in [-1, 0]
        dist_ratio = d_pat / d_sys; near miss (close to 1) → small penalty.
        Far miss (close to 0) → large penalty.
    """
    if hit:
        return float(np.clip(time_ratio, 0.0, 1.0))
    else:
        return float(np.clip(dist_ratio - 1.0, -1.0, 0.0))


# ----------------------------
# Tabular Q-learner
# ----------------------------
class TabularQLearner:
    """
    Tabular Q-learner over a 1D distance grid.

    State  : index of the current d in d_grid (0 .. n_states-1)
    Action : index of the next d to assign   (0 .. n_states-1) — any jump allowed
    Q-table: numpy array of shape (n_states, n_states)

    The next state after taking action a is simply a (we are now at d_grid[a]).
    """

    def __init__(self, cfg: QLearningConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        n = cfg.n_states
        self.d_grid = np.linspace(cfg.d_min, cfg.d_max, n)
        self.q_table = np.zeros((n, n), dtype=float)
        self.epsilon = cfg.epsilon_start

    def select_action(self, state: int) -> int:
        """Epsilon-greedy: random with prob epsilon, else argmax Q(state, .)."""
        if float(self.rng.uniform()) < self.epsilon:
            return int(self.rng.integers(0, self.cfg.n_states))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Standard Q-learning (Bellman) update."""
        td_target = reward + self.cfg.gamma * float(np.max(self.q_table[next_state]))
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.cfg.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)


# ----------------------------
# Main simulation loop
# ----------------------------
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

    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        dmin, dmax = derive_bounds_from_calibration(calibration_result, patient)
        cfg.d_min = dmin
        cfg.d_max = dmax
    cfg.d_min, cfg.d_max = cap_distance_bounds(patient, cfg.d_min, cfg.d_max)

    agent = TabularQLearner(cfg=cfg, rng=rng)

    counts = np.zeros((cfg.n_states, 1), dtype=int)

    logs: Dict[str, List] = {
        "trial": [],
        "state": [],
        "action": [],
        "d": [],
        "t": [],
        "hit": [],
        "reward": [],
        "epsilon": [],
        "time_ratio": [],
        "dist_ratio": [],
        "q_max": [],
    }

    # Start from the middle of the grid
    state = cfg.n_states // 2
    previous_hit = True

    for k in range(n_trials):
        action = agent.select_action(state)
        d_sys = float(agent.d_grid[action])
        t_sys = float(cfg.t_fixed)

        lvl = distance_level_from_patient_bins(patient, d_sys)
        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
            direction_bin=int(rng.integers(0, N_DIRECTIONS)),
        )

        hit = bool(outcome["hit"])
        previous_hit = hit
        time_ratio = float(outcome["time_ratio"])
        dist_ratio = float(outcome["dist_ratio"])

        reward = compute_reward(hit, dist_ratio, time_ratio)

        # Next state = action just taken (agent is now at d_grid[action])
        next_state = action
        agent.update(state, action, reward, next_state)
        agent.decay_epsilon()

        counts[action, 0] += 1

        logs["trial"].append(k)
        logs["state"].append(int(state))
        logs["action"].append(int(action))
        logs["d"].append(d_sys)
        logs["t"].append(t_sys)
        logs["hit"].append(int(hit))
        logs["reward"].append(float(reward))
        logs["epsilon"].append(float(agent.epsilon))
        logs["time_ratio"].append(time_ratio)
        logs["dist_ratio"].append(dist_ratio)
        logs["q_max"].append(float(np.max(agent.q_table[next_state])))

        state = next_state

    return logs, counts, patient
