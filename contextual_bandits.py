"""
Non-Contextual Beta-Bernoulli Thompson Sampling simulation for an AR rehab game
- 5 distance levels (10–80 cm), 5 time levels (1–7 s) => 25 actions
- Beta-Bernoulli Thompson Sampling (non-contextual) with multi-trial updates
- Uses PatientModel from patient_simulation.py with mode-switching hit/miss logic
- Graded reward (0.0, 0.25, 0.5, 0.75, 1.0) based on time_ratio and dist_ratio
  - Hits: rewarded by time_ratio (higher = more challenging)
  - Misses: rewarded by dist_ratio (higher = closer attempt)
- Visualizations included
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from patient_simulation_v2 import PatientModel

# -----------------------------
# 1) Define action space (25 items: 5 distance × 5 time)
# -----------------------------
DIST_MIN_M, DIST_MAX_M = 0.10, 0.80
TIME_MIN_S, TIME_MAX_S = 1.0, 7.0

NUM_DIST_BINS = 5
NUM_TIME_BINS = 5

def split_into_n(minv, maxv, n):
    """Return n closed intervals that partition [minv,maxv]."""
    bins = []
    for i in range(n):
        a = minv + i * (maxv - minv) / n
        b = minv + (i + 1) * (maxv - minv) / n
        bins.append((a, b))
    return bins

DIST_BINS = split_into_n(DIST_MIN_M, DIST_MAX_M, NUM_DIST_BINS)
TIME_BINS = split_into_n(TIME_MIN_S, TIME_MAX_S, NUM_TIME_BINS)

DIST_NAMES = [f"D{i}" for i in range(NUM_DIST_BINS)]
TIME_NAMES = [f"T{i}" for i in range(NUM_TIME_BINS)]

def sample_uniform_from_bin(bin_tuple, rng):
    lo, hi = bin_tuple
    return rng.uniform(lo, hi)

def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin + 1e-12)

@dataclass(frozen=True)
class Action:
    dist_level: int
    time_level: int
    dist_m: float
    time_s: float

def build_actions(rng):
    """Build all action combinations (not used in current implementation)."""
    actions = []
    for di in range(NUM_DIST_BINS):
        for ti in range(NUM_TIME_BINS):
            dist = sample_uniform_from_bin(DIST_BINS[di], rng)
            t = sample_uniform_from_bin(TIME_BINS[ti], rng)
            actions.append(Action(di, ti, dist, t))
    print(actions)
    return actions

# -----------------------------
# 2) Map distance levels to PatientModel difficulty levels
# -----------------------------
def map_distance_to_patient_level(dist_level):
    """
    Map our 5 distance levels to PatientModel's 8 difficulty levels.
    Use linear interpolation: dist_level 0-4 maps to patient_level 0-7
    """
    # Linear mapping from [0, 4] to [0, 7]
    patient_level = int(np.round(dist_level * 7.0 / 4.0))
    return np.clip(patient_level, 0, 7)

# -----------------------------
# 3) Graded Reward Function
# -----------------------------
def calculate_reward(outcome, hit_time_thresholds=None, miss_dist_thresholds=None):
    """
    Calculate 5-level graded reward based on time_ratio and dist_ratio.

    Parameters
    ----------
    outcome : dict
        Patient trial outcome with keys: hit, time_ratio, dist_ratio
    hit_time_thresholds : list of float, optional
        Thresholds for time_ratio when hit. Default: [0.8, 0.6, 0.4, 0.2]
        Maps to rewards: [1.0, 0.75, 0.5, 0.25, 0.0]
    miss_dist_thresholds : list of float, optional
        Thresholds for dist_ratio when miss. Default: [0.8, 0.6, 0.4, 0.2]
        Maps to rewards: [1.0, 0.75, 0.5, 0.25, 0.0]

    Returns
    -------
    float
        Reward in {0.0, 0.25, 0.5, 0.75, 1.0}
    """
    # Default thresholds (clinician-configurable)
    if hit_time_thresholds is None:
        hit_time_thresholds = [0.8, 0.6, 0.4, 0.2]
    if miss_dist_thresholds is None:
        miss_dist_thresholds = [0.8, 0.6, 0.4, 0.2]

    reward_levels = [1.0, 0.75, 0.5, 0.25, 0.0]

    hit = outcome["hit"]
    time_ratio = outcome["time_ratio"]
    dist_ratio = outcome["dist_ratio"]

    if hit:
        # HIT: Reward based on time_ratio (higher = used more time = more challenging)
        for i, threshold in enumerate(hit_time_thresholds):
            if time_ratio >= threshold:
                return reward_levels[i]
        return reward_levels[-1]  # Below all thresholds
    else:
        # MISS: Reward based on dist_ratio (higher = got closer = better attempt)
        for i, threshold in enumerate(miss_dist_thresholds):
            if dist_ratio >= threshold:
                return reward_levels[i]
        return reward_levels[-1]  # Below all thresholds

# -----------------------------
# 4) Beta-Bernoulli Thompson Sampling (Non-Contextual)
# -----------------------------
class BetaBernoulliTS:
    """
    Non-contextual Beta-Bernoulli Thompson Sampling with multi-trial updates.
    - Maintains a Beta(alpha_i, beta_i) distribution for each action i
    - Thompson sampling: sample theta_i ~ Beta(alpha_i, beta_i) for each action
      and pick argmax_i theta_i
    - Update: Converts graded reward to virtual trials
        reward 1.0 -> 4 successes, 0 failures
        reward 0.75 -> 3 successes, 1 failure
        reward 0.5 -> 2 successes, 2 failures
        reward 0.25 -> 1 success, 3 failures
        reward 0.0 -> 0 successes, 4 failures
    """

    def __init__(self, n_actions, virtual_trials=2, rng=None):
        self.n_actions = n_actions
        self.virtual_trials = virtual_trials
        self.rng = np.random.default_rng() if rng is None else rng

        # Initialize with uniform prior Beta(1, 1)
        self.alpha = np.ones(n_actions, dtype=float)
        self.beta = np.ones(n_actions, dtype=float)

    def sample_theta(self):
        """Sample success probability for each action from its Beta distribution."""
        return self.rng.beta(self.alpha, self.beta)

    def choose_action(self):
        """Sample theta for all actions and return argmax."""
        theta_samples = self.sample_theta()
        return int(np.argmax(theta_samples))

    def update(self, action_idx, reward):
        """
        Update Beta distribution for the chosen action using multi-trial approach.

        Parameters
        ----------
        action_idx : int
            Index of chosen action
        reward : float
            Graded reward in {0.0, 0.25, 0.5, 0.75, 1.0}
        """
        # Convert reward to number of successes out of virtual_trials
        n_successes = int(np.round(reward * self.virtual_trials))
        n_failures = self.virtual_trials - n_successes

        self.alpha[action_idx] += n_successes
        self.beta[action_idx] += n_failures

    def get_mean_estimates(self):
        """Return mean success probability estimate for each action."""
        return self.alpha / (self.alpha + self.beta)

# -----------------------------
# 5) Run simulation
# -----------------------------
def run_simulation(
    n_trials=100,
    seed=7,
    init_skill_est=0.5
):
    rng = np.random.default_rng(seed)

    # Note: actions are not pre-built; instead we choose action category then sample
    # Initialize Beta-Bernoulli Thompson Sampling with 25 actions (5 dist × 5 time)
    n_actions = NUM_DIST_BINS * NUM_TIME_BINS
    bandit = BetaBernoulliTS(n_actions=n_actions, rng=rng)
    patient = PatientModel(seed=seed)

    # Track previous hit for mode switching
    previous_hit = True

    # logs
    chosen_idx = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials, dtype=float)
    successes = np.zeros(n_trials, dtype=int)
    dist_m = np.zeros(n_trials, dtype=float)
    time_s = np.zeros(n_trials, dtype=float)
    dist_sys = np.zeros(n_trials, dtype=float)
    time_sys = np.zeros(n_trials, dtype=float)

    for t in range(n_trials):
        # Step 1: Choose action (distance_level, time_level) using Thompson Sampling
        a_idx = bandit.choose_action()
        dist_level = a_idx // NUM_TIME_BINS  # 0 to NUM_DIST_BINS-1
        time_level = a_idx % NUM_TIME_BINS   # 0 to NUM_TIME_BINS-1

        # Step 2: Sample random distance and time within the chosen action's bounds
        d_sys = sample_uniform_from_bin(DIST_BINS[dist_level], rng)
        t_sys = sample_uniform_from_bin(TIME_BINS[time_level], rng)

        # Step 3: Map distance level to PatientModel difficulty level
        patient_level = map_distance_to_patient_level(dist_level)

        # Step 4: Use PatientModel to determine hit/miss
        outcome = patient.sample_trial(
            t_sys=t_sys,
            d_sys=d_sys,
            distance_level=patient_level,
            previous_hit=previous_hit
        )

        # Extract hit/miss result
        hit = outcome["hit"]
        previous_hit = hit  # Update for next trial

        # Calculate graded reward based on time_ratio and dist_ratio
        reward = calculate_reward(outcome)

        # Update Beta-Bernoulli bandit with graded reward
        bandit.update(a_idx, reward)

        # logs
        chosen_idx[t] = a_idx
        rewards[t] = reward
        successes[t] = int(hit)
        dist_m[t] = outcome["d_pat"]  # actual distance achieved by patient
        time_s[t] = outcome["t_pat"]  # actual time taken by patient
        dist_sys[t] = d_sys  # system target distance
        time_sys[t] = t_sys  # system target time

    return {
        "chosen_idx": chosen_idx,
        "rewards": rewards,
        "successes": successes,
        "dist_m": dist_m,
        "time_s": time_s,
        "dist_sys": dist_sys,
        "time_sys": time_sys,
    }

# -----------------------------
# 6) Visualizations
# -----------------------------
def rolling_mean(x, w=25):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")

def plot_results(logs):
    chosen = logs["chosen_idx"]
    rewards = logs["rewards"]
    successes = logs["successes"]
    dist_m = logs["dist_m"]
    time_s = logs["time_s"]
    dist_sys = logs["dist_sys"]
    time_sys = logs["time_sys"]

    # 5x7 count heatmap
    counts = np.zeros((NUM_DIST_BINS, NUM_TIME_BINS), dtype=int)
    avg_success = np.zeros((NUM_DIST_BINS, NUM_TIME_BINS), dtype=float)
    avg_reward = np.zeros((NUM_DIST_BINS, NUM_TIME_BINS), dtype=float)
    n = np.zeros((NUM_DIST_BINS, NUM_TIME_BINS), dtype=int)

    for t, idx in enumerate(chosen):
        dist_level = int(idx // NUM_TIME_BINS)
        time_level = int(idx % NUM_TIME_BINS)
        counts[dist_level, time_level] += 1
        avg_success[dist_level, time_level] += successes[t]
        avg_reward[dist_level, time_level] += rewards[t]
        n[dist_level, time_level] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_success = np.where(n > 0, avg_success / n, 0.0)
        avg_reward = np.where(n > 0, avg_reward / n, 0.0)

    # Plot 1: action counts
    plt.figure(figsize=(12, 6))
    plt.imshow(counts, aspect="auto", cmap="Blues")
    plt.xticks(range(NUM_TIME_BINS), TIME_NAMES)
    plt.yticks(range(NUM_DIST_BINS), DIST_NAMES)
    plt.title("Action Selection Counts (distance x time)")
    plt.xlabel("Time level")
    plt.ylabel("Distance level")
    for i in range(NUM_DIST_BINS):
        for j in range(NUM_TIME_BINS):
            plt.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=8)
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.show()

    # Plot 2: average success rate per action
    plt.figure(figsize=(12, 6))
    plt.imshow(avg_success, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.xticks(range(NUM_TIME_BINS), TIME_NAMES)
    plt.yticks(range(NUM_DIST_BINS), DIST_NAMES)
    plt.title("Average Hit Rate by Action (distance x time)")
    plt.xlabel("Time level")
    plt.ylabel("Distance level")
    for i in range(NUM_DIST_BINS):
        for j in range(NUM_TIME_BINS):
            if n[i, j] > 0:
                plt.text(j, i, f"{avg_success[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(label="Hit Rate")
    plt.tight_layout()
    plt.show()

    # Plot 3: average reward per action
    plt.figure(figsize=(12, 6))
    plt.imshow(avg_reward, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.xticks(range(NUM_TIME_BINS), TIME_NAMES)
    plt.yticks(range(NUM_DIST_BINS), DIST_NAMES)
    plt.title("Average Graded Reward by Action (distance x time)")
    plt.xlabel("Time level")
    plt.ylabel("Distance level")
    for i in range(NUM_DIST_BINS):
        for j in range(NUM_TIME_BINS):
            if n[i, j] > 0:
                plt.text(j, i, f"{avg_reward[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color='white' if avg_reward[i, j] > 0.5 else 'black')
    plt.colorbar(label="Average Reward")
    plt.tight_layout()
    plt.show()

    # Plot 4: rolling success rate
    rs = rolling_mean(successes, w=25)
    rr = rolling_mean(rewards, w=25)
    x_s = np.arange(len(rs))
    x_r = np.arange(len(rr))

    plt.figure(figsize=(10, 5))
    plt.plot(x_s, rs, linewidth=2, label='Hit Rate')
    plt.title("Rolling Hit Rate (window=25)")
    plt.xlabel("Trial")
    plt.ylabel("Hit rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 5: rolling reward
    plt.figure(figsize=(10, 5))
    plt.plot(x_r, rr, linewidth=2, color='orange', label='Graded Reward')
    plt.title("Rolling Average Reward (window=25)")
    plt.xlabel("Trial")
    plt.ylabel("Average Reward")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7) Main
# -----------------------------
if __name__ == "__main__":
    logs = run_simulation(n_trials=50, seed=42)
    plot_results(logs)

    # Print summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Total trials: {len(logs['successes'])}")
    print(f"Overall hit rate: {np.mean(logs['successes']):.2%}")
    print(f"Overall avg reward: {np.mean(logs['rewards']):.3f}")
    print(f"\nPerformance by action (distance x time):")
    print(f"{'Action':12} {'Hit Rate':>10} {'Avg Reward':>12} {'Count':>8}")
    print("-" * 46)

    # Calculate statistics per action
    for dist_level in range(NUM_DIST_BINS):
        for time_level in range(NUM_TIME_BINS):
            a_idx = dist_level * NUM_TIME_BINS + time_level
            mask = logs["chosen_idx"] == a_idx
            if np.sum(mask) > 0:
                hit_rate = np.mean(logs["successes"][mask])
                avg_reward = np.mean(logs["rewards"][mask])
                count = np.sum(mask)
                action_name = f"{DIST_NAMES[dist_level]}x{TIME_NAMES[time_level]}"
                print(f"{action_name:12} {hit_rate:>9.2%} {avg_reward:>12.3f} {count:>8d}")