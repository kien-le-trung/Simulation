import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Tuple
from spatial_decomposition_TS import (
    spatial_decomposition,
    SpatialThompsonSampler,
    AngleBins
)


class PatientModel:
    """
    Simulates a patient with spatial-dependent reaching abilities.
    Better performance in certain directions (e.g., straight ahead, center).
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Patient abilities: better in center and forward directions
        # Azimuth preference: stronger near center (theta=0)
        # Elevation preference: stronger near center (phi=0)
        self.azimuth_preference_center = 0.0
        self.azimuth_preference_width = math.radians(25.0)
        self.elevation_preference_center = 0.0
        self.elevation_preference_width = math.radians(15.0)

    def get_base_success_prob(self, theta: float, phi: float) -> float:
        """
        Calculate base success probability based on spatial location.
        Higher in center, lower at edges.
        """
        # Gaussian-like preference
        az_factor = math.exp(-((theta - self.azimuth_preference_center)**2) /
                            (2 * self.azimuth_preference_width**2))
        el_factor = math.exp(-((phi - self.elevation_preference_center)**2) /
                            (2 * self.elevation_preference_width**2))

        # Base success rate: 0.3 to 0.9
        return 0.3 + 0.6 * az_factor * el_factor

    def simulate_trial(self, position: Tuple[float, float, float],
                      origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict:
        """
        Simulate a patient attempting to reach a target position.

        Returns outcome dict with:
        - hit: bool
        - time_ratio: float (if hit, higher = faster)
        - dist_ratio: float (if miss, higher = closer)
        """
        # Calculate angles from position
        dx = position[0] - origin[0]
        dy = position[1] - origin[1]
        dz = position[2] - origin[2]

        # Convert back to angles
        theta = math.atan2(dz, dx)  # azimuth
        phi = math.atan2(dy, math.sqrt(dx*dx + dz*dz))  # elevation

        # Get success probability
        success_prob = self.get_base_success_prob(theta, phi)

        # Simulate trial
        hit = self.rng.random() < success_prob

        if hit:
            # Time ratio: how quickly they reached (0-1, higher is better)
            # More variance when further from preference
            base_speed = success_prob
            time_ratio = min(1.0, max(0.0, base_speed + self.rng.gauss(0, 0.15)))
            dist_ratio = 0.0
        else:
            # Distance ratio: how close they got (0-1, higher is better)
            # Related to success probability
            base_closeness = success_prob * 0.7
            dist_ratio = min(1.0, max(0.0, base_closeness + self.rng.gauss(0, 0.15)))
            time_ratio = 0.0

        return {
            "hit": hit,
            "time_ratio": time_ratio,
            "dist_ratio": dist_ratio,
            "theta": theta,
            "phi": phi,
            "success_prob": success_prob
        }


def run_simulation(n_trials: int = 200, distance: float = 0.5, seed: int = 123):
    """
    Run a full simulation with Thompson sampling adaptation.

    Returns:
    - trials: list of trial data
    - sampler: the trained sampler
    """
    # Reset global sampler
    import spatial_decomposition_TS
    spatial_decomposition_TS._GLOBAL_SAMPLER = None

    patient = PatientModel(seed=seed + 1)
    origin = (0.0, 0.0, 0.0)

    trials = []

    # First trial: no outcome yet
    position = spatial_decomposition(distance, outcome=None, origin=origin, seed=seed)

    for trial_idx in range(n_trials):
        # Patient attempts the current position
        outcome = patient.simulate_trial(position, origin)

        # Record trial data
        trial_data = {
            'trial': trial_idx,
            'position': position,
            'outcome': outcome,
            'bin': spatial_decomposition_TS._GLOBAL_SAMPLER._last_bin if spatial_decomposition_TS._GLOBAL_SAMPLER else None
        }
        trials.append(trial_data)

        # Get next position (and update sampler with outcome)
        position = spatial_decomposition(distance, outcome=outcome, origin=origin, seed=seed)

    return trials, spatial_decomposition_TS._GLOBAL_SAMPLER


def plot_learning_curves(trials):
    """
    Plot success rate and reward over trials to show learning.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract metrics
    hit_rate = []
    rewards = []
    time_ratios = []
    dist_ratios = []
    success_probs = []

    window_size = 20

    for i in range(len(trials)):
        start_idx = max(0, i - window_size + 1)
        window = trials[start_idx:i+1]

        hits = sum(1 for t in window if t['outcome']['hit'])
        hit_rate.append(hits / len(window))

        # Calculate rewards using the same function
        from spatial_decomposition_TS import calculate_reward
        window_rewards = [calculate_reward(t['outcome']) for t in window]
        rewards.append(np.mean(window_rewards))

        # Track time and distance ratios
        if trials[i]['outcome']['hit']:
            time_ratios.append(trials[i]['outcome']['time_ratio'])
        if not trials[i]['outcome']['hit']:
            dist_ratios.append(trials[i]['outcome']['dist_ratio'])

        success_probs.append(trials[i]['outcome']['success_prob'])

    # Plot 1: Hit rate over time
    axes[0, 0].plot(hit_rate, linewidth=2, color='steelblue')
    axes[0, 0].set_xlabel('Trial', fontsize=11)
    axes[0, 0].set_ylabel(f'Success Rate (window={window_size})', fontsize=11)
    axes[0, 0].set_title('Learning Curve: Success Rate', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Plot 2: Average reward over time
    axes[0, 1].plot(rewards, linewidth=2, color='forestgreen')
    axes[0, 1].set_xlabel('Trial', fontsize=11)
    axes[0, 1].set_ylabel(f'Average Reward (window={window_size})', fontsize=11)
    axes[0, 1].set_title('Learning Curve: Reward', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Plot 3: Success probability (ground truth)
    axes[1, 0].plot(success_probs, linewidth=1.5, color='coral', alpha=0.6, label='Actual')
    axes[1, 0].plot(np.convolve(success_probs, np.ones(window_size)/window_size, mode='same'),
                    linewidth=2, color='red', label=f'Moving Avg ({window_size})')
    axes[1, 0].set_xlabel('Trial', fontsize=11)
    axes[1, 0].set_ylabel('True Success Probability', fontsize=11)
    axes[1, 0].set_title('Patient Ability at Selected Positions', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])

    # Plot 4: Cumulative hits vs misses
    cumulative_hits = np.cumsum([1 if t['outcome']['hit'] else 0 for t in trials])
    cumulative_misses = np.cumsum([0 if t['outcome']['hit'] else 1 for t in trials])
    axes[1, 1].plot(cumulative_hits, linewidth=2, color='green', label='Hits')
    axes[1, 1].plot(cumulative_misses, linewidth=2, color='red', label='Misses')
    axes[1, 1].set_xlabel('Trial', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Count', fontsize=11)
    axes[1, 1].set_title('Cumulative Hits vs Misses', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    return fig


def plot_spatial_heatmap(sampler: SpatialThompsonSampler, patient: PatientModel):
    """
    Plot heatmap of learned preferences vs true patient abilities.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bins = sampler.bins
    n_az, n_el = bins.n_az, bins.n_el

    # Calculate mean of Beta distributions (alpha / (alpha + beta))
    learned_probs = np.zeros((n_el, n_az))
    for i in range(n_az):
        for j in range(n_el):
            a = sampler.alpha[i][j]
            b = sampler.beta[i][j]
            learned_probs[j, i] = a / (a + b)

    # Calculate true success probabilities at bin centers
    true_probs = np.zeros((n_el, n_az))
    for i in range(n_az):
        for j in range(n_el):
            # Get bin center
            az_w = (bins.az_max - bins.az_min) / n_az
            el_w = (bins.el_max - bins.el_min) / n_el
            theta = bins.az_min + (i + 0.5) * az_w
            phi = bins.el_min + (j + 0.5) * el_w
            true_probs[j, i] = patient.get_base_success_prob(theta, phi)

    # Calculate number of observations per bin
    observations = np.zeros((n_el, n_az))
    for i in range(n_az):
        for j in range(n_el):
            # Total observations = (alpha + beta - 2) since we start at (1,1)
            observations[j, i] = sampler.alpha[i][j] + sampler.beta[i][j] - 2

    # Create azimuth and elevation labels in degrees
    az_labels = [f"{math.degrees(bins.az_min + (i+0.5)*(bins.az_max-bins.az_min)/n_az):.0f}°"
                 for i in range(n_az)]
    el_labels = [f"{math.degrees(bins.el_min + (j+0.5)*(bins.el_max-bins.el_min)/n_el):.0f}°"
                 for j in range(n_el)]

    # Plot 1: Learned probabilities
    im1 = axes[0].imshow(learned_probs, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Learned Success Probability\n(Thompson Sampler)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Azimuth (degrees)', fontsize=11)
    axes[0].set_ylabel('Elevation (degrees)', fontsize=11)
    axes[0].set_xticks(range(n_az))
    axes[0].set_xticklabels(az_labels, rotation=45, ha='right')
    axes[0].set_yticks(range(n_el))
    axes[0].set_yticklabels(el_labels)
    plt.colorbar(im1, ax=axes[0], label='Success Probability')

    # Plot 2: True probabilities
    im2 = axes[1].imshow(true_probs, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('True Patient Ability\n(Ground Truth)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Azimuth (degrees)', fontsize=11)
    axes[1].set_ylabel('Elevation (degrees)', fontsize=11)
    axes[1].set_xticks(range(n_az))
    axes[1].set_xticklabels(az_labels, rotation=45, ha='right')
    axes[1].set_yticks(range(n_el))
    axes[1].set_yticklabels(el_labels)
    plt.colorbar(im2, ax=axes[1], label='Success Probability')

    # Plot 3: Number of observations
    im3 = axes[2].imshow(observations, cmap='Blues', aspect='auto')
    axes[2].set_title('Exploration Pattern\n(# of Trials per Bin)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Azimuth (degrees)', fontsize=11)
    axes[2].set_ylabel('Elevation (degrees)', fontsize=11)
    axes[2].set_xticks(range(n_az))
    axes[2].set_xticklabels(az_labels, rotation=45, ha='right')
    axes[2].set_yticks(range(n_el))
    axes[2].set_yticklabels(el_labels)
    plt.colorbar(im3, ax=axes[2], label='# Observations')

    # Annotate with values
    for i in range(n_az):
        for j in range(n_el):
            axes[2].text(i, j, f'{int(observations[j, i])}',
                        ha='center', va='center', fontsize=8, color='white' if observations[j, i] > observations.max()/2 else 'black')

    plt.tight_layout()
    return fig


def plot_beta_distributions(sampler: SpatialThompsonSampler, n_samples: int = 6):
    """
    Plot Beta distributions for a sample of bins to show uncertainty.
    """
    bins = sampler.bins
    n_az, n_el = bins.n_az, bins.n_el

    # Select bins with different amounts of data
    # Find bins with: high alpha, high beta, balanced, low data
    all_bins = []
    for i in range(n_az):
        for j in range(n_el):
            a = sampler.alpha[i][j]
            b = sampler.beta[i][j]
            all_bins.append((i, j, a, b, a+b))

    # Sort by total observations
    all_bins.sort(key=lambda x: x[4], reverse=True)

    # Select diverse bins
    selected = [
        all_bins[0],  # Most observed
        all_bins[1],
        all_bins[2],
        all_bins[3],
        all_bins[-2],
        all_bins[-1],  # Least observed
    ][:n_samples]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    x = np.linspace(0, 1, 200)

    for idx, (i, j, a, b, total) in enumerate(selected):
        # Calculate Beta PDF
        from scipy.stats import beta as beta_dist
        y = beta_dist.pdf(x, a, b)

        # Get bin center angles
        az_w = (bins.az_max - bins.az_min) / n_az
        el_w = (bins.el_max - bins.el_min) / n_el
        theta = bins.az_min + (i + 0.5) * az_w
        phi = bins.el_min + (j + 0.5) * el_w

        axes[idx].plot(x, y, linewidth=2.5, color='steelblue')
        axes[idx].fill_between(x, y, alpha=0.3, color='steelblue')
        axes[idx].axvline(a/(a+b), color='red', linestyle='--', linewidth=2, label=f'Mean={a/(a+b):.2f}')
        axes[idx].set_xlabel('Success Probability', fontsize=10)
        axes[idx].set_ylabel('Density', fontsize=10)
        axes[idx].set_title(f'Bin ({i},{j}): Az={math.degrees(theta):.0f}°, El={math.degrees(phi):.0f}°\n' +
                           f'α={a:.1f}, β={b:.1f}, n={int(total-2)}', fontsize=10)
        axes[idx].grid(alpha=0.3)
        axes[idx].legend(fontsize=9)
        axes[idx].set_ylim(bottom=0)

    plt.suptitle('Beta Posterior Distributions for Selected Bins', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_3d_positions(trials, patient: PatientModel, n_show: int = 100):
    """
    3D scatter plot of target positions colored by success/failure.
    """
    fig = plt.figure(figsize=(14, 6))

    # Plot 1: First n_show trials
    ax1 = fig.add_subplot(121, projection='3d')

    early_trials = trials[:n_show]
    for t in early_trials:
        pos = t['position']
        color = 'green' if t['outcome']['hit'] else 'red'
        alpha = 0.6
        ax1.scatter(pos[0], pos[1], pos[2], c=color, alpha=alpha, s=50)

    ax1.set_xlabel('X (Right)', fontsize=10)
    ax1.set_ylabel('Y (Up)', fontsize=10)
    ax1.set_zlabel('Z (Forward)', fontsize=10)
    ax1.set_title(f'Target Positions: First {n_show} Trials', fontsize=12, fontweight='bold')
    ax1.legend(['Hit', 'Miss'])

    # Plot 2: Last n_show trials
    ax2 = fig.add_subplot(122, projection='3d')

    late_trials = trials[-n_show:]
    for t in late_trials:
        pos = t['position']
        color = 'green' if t['outcome']['hit'] else 'red'
        alpha = 0.6
        ax2.scatter(pos[0], pos[1], pos[2], c=color, alpha=alpha, s=50)

    ax2.set_xlabel('X (Right)', fontsize=10)
    ax2.set_ylabel('Y (Up)', fontsize=10)
    ax2.set_zlabel('Z (Forward)', fontsize=10)
    ax2.set_title(f'Target Positions: Last {n_show} Trials', fontsize=12, fontweight='bold')
    ax2.legend(['Hit', 'Miss'])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Starting Thompson Sampling Simulation...")
    print("=" * 60)

    # Run simulation
    n_trials = 50
    distance = np.random.uniform(0.1, 0.8)
    seed = 123

    print(f"Running {n_trials} trials at distance {distance}m...")
    trials, sampler = run_simulation(n_trials=n_trials, distance=distance, seed=seed)

    # Calculate final statistics
    total_hits = sum(1 for t in trials if t['outcome']['hit'])
    hit_rate = total_hits / len(trials)

    last_50 = trials[-50:]
    recent_hits = sum(1 for t in last_50 if t['outcome']['hit'])
    recent_hit_rate = recent_hits / len(last_50)

    print(f"\nSimulation Complete!")
    print(f"Overall success rate: {hit_rate:.2%}")
    print(f"Recent success rate (last 50 trials): {recent_hit_rate:.2%}")
    print(f"Improvement: {recent_hit_rate - hit_rate:+.2%}")
    print("=" * 60)

    # Create patient model for comparison plots
    patient = PatientModel(seed=seed + 1)

    # Generate all plots
    print("\nGenerating visualizations...")

    print("  [1/3] Learning curves...")
    fig1 = plot_learning_curves(trials)
    fig1.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    print("        Saved: learning_curves.png")

    print("  [2/3] Spatial heatmaps...")
    fig2 = plot_spatial_heatmap(sampler, patient)
    fig2.savefig('spatial_heatmap.png', dpi=150, bbox_inches='tight')
    print("        Saved: spatial_heatmap.png")

    print("  [3/3] Beta distributions...")
    fig3 = plot_beta_distributions(sampler)
    fig3.savefig('beta_distributions.png', dpi=150, bbox_inches='tight')
    print("        Saved: beta_distributions.png")

    print("\nAll visualizations saved!")
    print("=" * 60)

    plt.show()
