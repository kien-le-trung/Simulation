"""
Simulation loop integrating:
  - controller.py: RatioController adjusts t_sys and d_sys
  - patient_simulation.py: PatientModel samples t_pat and d_pat
  - patient_class.py: Patient tracks long-term categorical distribution
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from controller import RatioController
from patient_simulation import PatientModel
from patient_class import Patient


class Simulation:
    """
    Main simulation loop that coordinates:
      1. Controller sets target t_sys, d_sys
      2. PatientModel samples actual patient response t_pat, d_pat
      3. Patient class logs outcomes and updates categorical distribution
      4. Controller updates based on outcome
    """

    def __init__(
        self,
        # PatientModel parameters
        t_levels=(1., 2., 3., 4., 5., 6., 7., 8.),
        k_d_per_sec=0.25,
        sigma_t=0.35,
        sigma_d=0.08,
        patient_seed=7,

        # RatioController parameters
        beta_t=0.1,
        min_lr_t=0.05,
        min_lr_d=0.05,
        max_lr_d=1.5,
        t_bounds=(0, 8.0),
        d_bounds=(0.1, 2.5),

        # Patient class parameters
        distance_levels=range(8),  # Will be derived from t_levels
        time_min=1.0,
        time_max=8.0,
        num_time_bins=20,

        # Bayesian update parameters
        block_size=5,
        update_frequency=5,  # Update every N trials
    ):
        # Initialize patient simulation model
        self.patient_model = PatientModel(
            t_levels=t_levels,
            k_d_per_sec=k_d_per_sec,
            sigma_t=sigma_t,
            sigma_d=sigma_d,
            seed=patient_seed
        )

        # Initialize controller
        self.controllers = {}
        for distance_level in distance_levels:
            self.controllers[distance_level] = RatioController(
                beta_t=beta_t,
                min_lr_t=min_lr_t,
                min_lr_d=min_lr_d,
                max_lr_d=max_lr_d,
                t_bounds=t_bounds,
                d_bounds=d_bounds
            )

        self.patient = Patient(
            distance_levels=distance_levels,
            time_min=time_min,
            time_max=time_max,
            num_time_bins=num_time_bins
        )

        # Store parameters
        self.block_size = block_size
        self.update_frequency = update_frequency
        self.sigma_t = sigma_t
        self.sigma_d = sigma_d

        # Trial counter
        self.trial_count = 0

        # History for analysis
        self.history_by_distance = {}
        for distance_level in distance_levels:
            self.history_by_distance[distance_level] = {
                't_sys': [],
                'd_sys': [],
                't_pat': [],
                'd_pat': [],
                'hit': [],
                'time_ratio': [],
                'dist_ratio': []
            }

    def map_distance_to_level(self, d_sys):
        """
        Map continuous d_sys to discrete distance level for Patient class.
        Uses the same bucketing logic as PatientModel.
        """
        idx = self.patient_model._bucket_by_distance(d_sys)
        return self.patient.distance_levels[idx]

    def run_trial(self, t_sys, d_sys, distance_level, previous_hit=True):
        """
        Execute a single trial:
          1. PatientModel samples outcome based on t_sys, d_sys
          2. Patient class logs the outcome
          3. Controller updates t_sys, d_sys

        Parameters
        ----------
        previous_hit : bool
            Whether the previous trial for this distance level was a hit
        """
        # Sample patient response
        outcome = self.patient_model.sample_trial(t_sys, d_sys, distance_level, previous_hit)

        # Controller update
        t_sys, d_sys = self.controllers[distance_level].update(
            self.patient_model,
            t_sys,
            d_sys,
            outcome,
        )

        # Log to patient class (only on hits for meaningful time data)
        if outcome['hit']:
            self.patient.map_time_to_bin(distance_level, outcome['t_pat'])

        # Update categorical distribution periodically
        if (self.trial_count + 1) % self.update_frequency == 0:
            self.patient.bayesian_update(
                distance_level,
                block_size=self.block_size,
                use_last_block_only=True
            )

        # Record history
        self.history_by_distance[distance_level]['t_sys'].append(t_sys)
        self.history_by_distance[distance_level]['d_sys'].append(d_sys)
        self.history_by_distance[distance_level]['t_pat'].append(outcome['t_pat'])
        self.history_by_distance[distance_level]['d_pat'].append(outcome['d_pat'])
        self.history_by_distance[distance_level]['hit'].append(outcome['hit'])
        self.history_by_distance[distance_level]['time_ratio'].append(outcome['time_ratio'])
        self.history_by_distance[distance_level]['dist_ratio'].append(outcome['dist_ratio'])

        # Increment trial counter
        self.trial_count += 1

        return outcome

    def run(self, num_trials, verbose=False, print_every=10):
        """
        Run the simulation for a specified number of trials.

        Parameters
        ----------
        num_trials : int
            Number of trials to run
        verbose : bool
            Whether to print progress information
        print_every : int
            Print frequency (every N trials) if verbose=True

        Returns
        -------
        history : dict
            Dictionary containing trial history
        """
        for i in range(num_trials):
            for distance_level in range(8):
                # If nothing in history, then initialize t_sys, d_sys
                if len(self.history_by_distance[distance_level]["t_sys"]) == 0:
                    t_sys = self.patient_model.t_means[distance_level]
                    d_sys = self.patient_model.d_means[distance_level]
                    previous_hit = True  # Start in time-based mode
                # Otherwise fetch the last value in history
                else:
                    t_sys = self.history_by_distance[distance_level]["t_sys"][-1]
                    d_sys = self.history_by_distance[distance_level]["d_sys"][-1]
                    previous_hit = self.history_by_distance[distance_level]["hit"][-1]
                # 40 trials per session. After each session, re-initialize t_sys and d_dys
                if i % 39 == 0 and i != 0:
                    d_sys = self.patient_model.d_means[distance_level]
                    # Find the time bin where cumulative probability reaches 0.67 (1 standard dev from the mean)
                    categorical_probs = self.patient.categorical_probs[distance_level]
                    cumulative_probs = np.cumsum(categorical_probs)
                    target_idx = np.searchsorted(cumulative_probs, 0.67)
                    # Map the bin index back to a time value
                    time_bins = np.linspace(self.patient.time_min, self.patient.time_max, self.patient.num_time_bins)
                    if target_idx < len(time_bins):
                        t_sys = time_bins[target_idx]
                    else:
                        t_sys = time_bins[-1]
                    print(f"Re-initializing t_sys to {t_sys:.3f} and d_sys to {d_sys:.3f} at trial {i+1} for distance level {distance_level}")
                outcome = self.run_trial(t_sys, d_sys, distance_level, previous_hit)

            if verbose and (i + 1) % print_every == 0:
                hit_str = "HIT" if outcome['hit'] else "MISS"
                print(f"Trial {i+1:4d}: {hit_str} | "
                      f"t_sys={t_sys:.3f}, d_sys={d_sys:.3f} | "
                      f"t_pat={outcome['t_pat']:.3f}, d_pat={outcome['d_pat']:.3f} | "
                      f"time_ratio={outcome['time_ratio']:.3f}, dist_ratio={outcome['dist_ratio']:.3f}")

        return self.history_by_distance

    def reset(self):
        """Reset the simulation to initial state."""
        self.controller.reset()
        self.trial_count = 0

        # Reset patient class
        self.patient = Patient(
            distance_levels=self.patient.distance_levels,
            time_min=self.patient.time_min,
            time_max=self.patient.time_max,
            num_time_bins=self.patient.num_time_bins
        )

        # Reset targets
        self.t_sys = np.mean([self.controller.t_min, self.controller.t_max])
        self.d_sys = np.mean([self.controller.d_min, self.controller.d_max])

        # Clear history
        self.history = {
            't_sys': [],
            'd_sys': [],
            't_pat': [],
            'd_pat': [],
            'hit': [],
            'time_ratio': [],
            'dist_ratio': [],
            'distance_level': []
        }

    def get_summary(self):
        """Get summary statistics of the simulation."""
        if self.trial_count == 0:
            return "No trials run yet."

        hits = np.sum(self.history['hit'])
        hit_rate = hits / self.trial_count

        summary = f"""
Simulation Summary
==================
Total Trials: {self.trial_count}
Hits: {hits} ({hit_rate*100:.1f}%)
Misses: {self.trial_count - hits} ({(1-hit_rate)*100:.1f}%)

System Targets (final):
  t_sys: {self.t_sys:.3f}
  d_sys: {self.d_sys:.3f}

Average Patient Response:
  t_pat: {np.mean(self.history['t_pat']):.3f} ± {np.std(self.history['t_pat']):.3f}
  d_pat: {np.mean(self.history['d_pat']):.3f} ± {np.std(self.history['d_pat']):.3f}

Average Ratios:
  time_ratio: {np.mean(self.history['time_ratio']):.3f}
  dist_ratio: {np.mean(self.history['dist_ratio']):.3f}
"""
        return summary


# Example usage
if __name__ == "__main__":
    # Create simulation
    sim = Simulation(
        t_levels=(1., 2., 3., 4., 5., 6., 7., 8.),
        k_d_per_sec=0.25,
        sigma_t=0.35,
        sigma_d=0.08,
        patient_seed=42
    )

    # Run simulation
    print("Running simulation for 120 trials...\n")
    history = sim.run(num_trials=1200, verbose=True, print_every=20)

    # Export history to CSV
    rows = []
    for distance_level, data in history.items():
        for i in range(len(data['t_sys'])):
            rows.append({
                'distance_level': distance_level,
                't_sys': data['t_sys'][i],
                'd_sys': data['d_sys'][i],
                't_pat': data['t_pat'][i],
                'd_pat': data['d_pat'][i],
                'hit': data['hit'][i],
                'time_ratio': data['time_ratio'][i],
                'dist_ratio': data['dist_ratio'][i]
            })
    df = pd.DataFrame(rows)
    df = df.replace({ True: 1, False: 0 })  # Convert boolean to int for CSV
    df.to_csv('simulation_history.csv', index=False)

    # Visualize sim.patient.categorical_probs as a heatmap
    probs = sim.patient.categorical_probs  # shape: (n_dist_levels, n_time_bins)
    np.savetxt('categorical_probs.csv', probs, delimiter=',', fmt='%f')
    dist_levels = list(sim.patient.distance_levels)
    n_dist = probs.shape[0]
    n_bins = probs.shape[1]
    time_bin_edges = np.linspace(sim.patient.time_min, sim.patient.time_max, n_bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Show heatmap with distance levels on y-axis and time on x-axis
    im = ax.imshow(probs, aspect='auto', origin='lower', cmap='viridis',
                   extent=[time_bin_edges[0], time_bin_edges[-1], -0.5, n_dist - 0.5])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance level (index)')
    ax.set_yticks(np.arange(n_dist))
    ax.set_yticklabels(dist_levels)
    ax.set_title('Patient categorical_probs (distance level × time bin)')
    plt.tight_layout()
    plt.show()