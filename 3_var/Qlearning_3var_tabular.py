from __future__ import annotations

import importlib.util
from dataclasses import dataclass
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

# Action indices
D_UP    = 0   # increase d (harder ROM demand)
D_DOWN  = 1   # decrease d (easier ROM demand)
T_UP    = 2   # increase t (more time = easier)
T_DOWN  = 3   # decrease t (less time = harder)
DIR_UP  = 4   # cycle direction index up
DIR_DOWN = 5  # cycle direction index down
N_ACTIONS = 6


@dataclass
class QLearningConfig:
    d_min: float = 0.10
    d_max: float = 0.80
    t_min: float = 1.0
    t_max: float = 7.0
    n_d: int = 8          # number of d grid points
    n_t: int = 8          # number of t grid points
    n_dir: int = N_DIRECTIONS
    alpha: float = 0.20
    gamma: float = 0.92
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


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
    ABS_D_MIN, ABS_D_MAX = 0.05, 1.5
    ABS_T_MIN, ABS_T_MAX = 0.15, 15.0
    defaults = QLearningConfig()

    if not calibration_result:
        return defaults.d_min, defaults.d_max, defaults.t_min, defaults.t_max

    trials = calibration_result.get("trials", [])
    speeds = []
    for tr in trials:
        hit = tr.get("hit", tr.get("reached", False))
        t_pat = float(tr.get("t_pat_obs", tr.get("t_pat", 0)))
        d_sys = float(tr.get("d_sys", 0))
        if hit and t_pat > 0.01 and d_sys > 0.01:
            speeds.append(d_sys / t_pat)

    if len(speeds) < 3:
        return defaults.d_min, defaults.d_max, defaults.t_min, defaults.t_max

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
# Reward (same design as 1var/2var tabular)
# ----------------------------
def compute_reward(hit: bool, dist_ratio: float, time_ratio: float) -> float:
    """
    HIT:  r = time_ratio in [0, 1]
        Barely-made-it hits (time_ratio ≈ 1) score highest, nudging the agent
        toward challenging but achievable (d, t, dir) combinations.

    MISS: r = dist_ratio - 1 in [-1, 0]
        Near misses (dist_ratio ≈ 1) receive small penalties; far misses
        (dist_ratio ≈ 0) receive large penalties, discouraging reckless overreach.
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
    Tabular Q-learner over a 3D (d, t, direction) lattice.

    State  : (d_idx, t_idx, dir_idx) — position on n_d × n_t × n_dir grid
    Actions: 6 incremental moves — D_UP, D_DOWN, T_UP, T_DOWN, DIR_UP, DIR_DOWN
             Moves that would leave the grid are clipped to the boundary.
             Direction wraps cyclically (0 → n_dir-1 and n_dir-1 → 0) since
             directions are nominal, not ordered — cycling is more natural than clamping.
    Q-table: numpy array of shape (n_d, n_t, n_dir, N_ACTIONS)
    """

    def __init__(self, cfg: QLearningConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.d_grid = np.linspace(cfg.d_min, cfg.d_max, cfg.n_d)
        self.t_grid = np.linspace(cfg.t_min, cfg.t_max, cfg.n_t)
        self.q_table = np.zeros((cfg.n_d, cfg.n_t, cfg.n_dir, N_ACTIONS), dtype=float)
        self.epsilon = cfg.epsilon_start

    def next_state(
        self, d_idx: int, t_idx: int, dir_idx: int, action: int
    ) -> Tuple[int, int, int]:
        """Apply action and return resulting (d_idx, t_idx, dir_idx)."""
        if action == D_UP:
            return min(d_idx + 1, self.cfg.n_d - 1), t_idx, dir_idx
        if action == D_DOWN:
            return max(d_idx - 1, 0), t_idx, dir_idx
        if action == T_UP:
            return d_idx, min(t_idx + 1, self.cfg.n_t - 1), dir_idx
        if action == T_DOWN:
            return d_idx, max(t_idx - 1, 0), dir_idx
        if action == DIR_UP:
            return d_idx, t_idx, (dir_idx + 1) % self.cfg.n_dir
        # DIR_DOWN
        return d_idx, t_idx, (dir_idx - 1) % self.cfg.n_dir

    def select_action(self, d_idx: int, t_idx: int, dir_idx: int) -> int:
        """Epsilon-greedy over the 6 local actions."""
        if float(self.rng.uniform()) < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(self.q_table[d_idx, t_idx, dir_idx]))

    def update(
        self,
        d_idx: int, t_idx: int, dir_idx: int,
        action: int,
        reward: float,
        nd_idx: int, nt_idx: int, ndir_idx: int,
    ) -> None:
        """Standard Q-learning (Bellman) update."""
        td_target = reward + self.cfg.gamma * float(
            np.max(self.q_table[nd_idx, nt_idx, ndir_idx])
        )
        td_error = td_target - self.q_table[d_idx, t_idx, dir_idx, action]
        self.q_table[d_idx, t_idx, dir_idx, action] += self.cfg.alpha * td_error

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
):
    cfg = cfg or QLearningConfig()
    rng = np.random.default_rng(seed)
    patient = patient or PatientModel(seed=seed)

    if calibration:
        calibration_result = patient.calibration()
        apply_calibration_priors(patient, calibration_result)
        dmin, dmax, tmin, tmax = derive_bounds_from_calibration(calibration_result, patient)
        cfg.d_min = dmin
        cfg.d_max = dmax
        cfg.t_min = tmin
        cfg.t_max = tmax
    cfg.d_min, cfg.d_max = cap_distance_bounds(patient, cfg.d_min, cfg.d_max)

    agent = TabularQLearner(cfg=cfg, rng=rng)

    counts = np.zeros((cfg.n_d, cfg.n_t), dtype=int)

    logs: Dict[str, List] = {
        "trial": [],
        "d_idx": [],
        "t_idx": [],
        "dir_idx": [],
        "action": [],
        "d": [],
        "t": [],
        "direction": [],
        "hit": [],
        "reward": [],
        "epsilon": [],
        "time_ratio": [],
        "dist_ratio": [],
        "q_max": [],
    }

    # Start from the centre of the grid
    d_idx = cfg.n_d // 2
    t_idx = cfg.n_t // 2
    dir_idx = cfg.n_dir // 2
    previous_hit = True

    for k in range(n_trials):
        action = agent.select_action(d_idx, t_idx, dir_idx)
        nd_idx, nt_idx, ndir_idx = agent.next_state(d_idx, t_idx, dir_idx, action)

        d_sys = float(agent.d_grid[nd_idx])
        t_sys = float(agent.t_grid[nt_idx])
        direction = int(ndir_idx)

        lvl = distance_level_from_patient_bins(patient, d_sys)
        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=lvl,
            previous_hit=previous_hit,
            direction_bin=direction,
        )

        hit = bool(outcome["hit"])
        previous_hit = hit
        time_ratio = float(outcome["time_ratio"])
        dist_ratio = float(outcome["dist_ratio"])

        reward = compute_reward(hit, dist_ratio, time_ratio)

        agent.update(d_idx, t_idx, dir_idx, action, reward, nd_idx, nt_idx, ndir_idx)
        agent.decay_epsilon()

        counts[nd_idx, nt_idx] += 1

        logs["trial"].append(k)
        logs["d_idx"].append(int(nd_idx))
        logs["t_idx"].append(int(nt_idx))
        logs["dir_idx"].append(int(ndir_idx))
        logs["action"].append(int(action))
        logs["d"].append(d_sys)
        logs["t"].append(t_sys)
        logs["direction"].append(direction)
        logs["hit"].append(int(hit))
        logs["reward"].append(float(reward))
        logs["epsilon"].append(float(agent.epsilon))
        logs["time_ratio"].append(time_ratio)
        logs["dist_ratio"].append(dist_ratio)
        logs["q_max"].append(float(np.max(agent.q_table[nd_idx, nt_idx, ndir_idx])))

        d_idx, t_idx, dir_idx = nd_idx, nt_idx, ndir_idx

    return logs, counts, patient
