from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Optional
import numpy as np


@dataclass
class Patient:
    """
    Patient model with:
      - categorical_probs: 8 x 10 matrix, P(time_bin | distance)
      - time_log: dict[distance_key] -> list of time_bin indices
      - dirichlet_alpha: 10 x 8 matrix (rows = time bins, cols = distances)
    """
    distance_levels: Sequence[float] = field(default_factory=lambda: list(range(8)))
    time_min: float = 1.0
    time_max: float = 8.0
    num_time_bins: int = 20

    # internal fields (initialized in __post_init__)
    time_bins: np.ndarray = field(init=False)          # shape (num_time_bins,)
    categorical_probs: np.ndarray = field(init=False)  # shape (num_distances, num_time_bins)
    dirichlet_alpha: np.ndarray = field(init=False)    # shape (num_time_bins, num_distances)
    time_log: Dict[float, List[int]] = field(init=False)

    def __post_init__(self):
        self.distance_levels = list(self.distance_levels)
        num_distances = len(self.distance_levels)

        # Time bin centers between time_min and time_max
        self.time_bins = np.linspace(self.time_min, self.time_max, self.num_time_bins)

        # --- Categorical distribution matrix: 8 x 10, uniform per distance ---
        # Rows: distances, Columns: time bins
        self.categorical_probs = np.full(
            (num_distances, self.num_time_bins),
            1.0 / self.num_time_bins,
            dtype=float
        )

        # --- Dirichlet distribution matrix: 10 x n (n = num_distances), uniform prior ---
        # Rows: time bins, Columns: distances
        # Each column is alpha vector for Dirichlet prior over 10 time bins at that distance
        self.dirichlet_alpha = np.ones(
            (self.num_time_bins, num_distances),
            dtype=float
        )

        # --- Time log: key = distance value, value = list of time-bin indices ---
        self.time_log = {d: [] for d in self.distance_levels}

    # ---------- Utility methods ----------

    def _get_distance_index(self, distance: float) -> int:
        """
        Map a distance value in distance_levels to its index.
        Raises ValueError if distance is not in distance_levels.
        """
        try:
            return self.distance_levels.index(distance)
        except ValueError:
            raise ValueError(f"Distance {distance} not found in distance_levels {self.distance_levels}")

    def get_closest_distance_level(self, continuous_distance: float) -> float:
        """
        Map a continuous distance value to the closest discrete distance level.
        Useful when integrating with continuous distance values from other components.

        Parameters
        ----------
        continuous_distance : float
            Any continuous distance value

        Returns
        -------
        float
            The closest discrete distance level from distance_levels
        """
        distances_array = np.array(self.distance_levels)
        idx = int(np.argmin(np.abs(distances_array - continuous_distance)))
        return self.distance_levels[idx]

    # ---------- Core API ----------

    def map_time_to_bin(self, distance: float, t: float) -> int:
        """
        Map continuous time t to the index of the closest time bin.
        Append that time-bin index to the log for this distance.

        Returns
        -------
        bin_idx : int
            Index in [0, num_time_bins - 1] of the closest bin.
        """
        # Clip t to [time_min, time_max] for safety
        t_clipped = float(np.clip(t, self.time_min, self.time_max))

        # Find the closest bin center
        bin_idx = int(np.argmin(np.abs(self.time_bins - t_clipped)))

        # Log the bin index for this distance
        if distance not in self.time_log:
            # If user passes a new distance not in distance_levels, we could choose to add it,
            # but for now we keep it strict.
            raise ValueError(f"Distance {distance} not found in time_log keys. "
                             f"Known distances: {list(self.time_log.keys())}")

        self.time_log[distance].append(bin_idx)
        return bin_idx

    def bayesian_update(
        self,
        distance: float,
        block_size: int = 5,
        use_last_block_only: bool = True,
    ) -> None:
        """
        Bayesian update of Dirichlet and hence categorical probabilities for a given distance.

        Parameters
        ----------
        distance : float
            The discrete distance (must be in distance_levels).
        block_size : int, optional
            Number of recent trials to use for the update (default = 5).
        use_last_block_only : bool, optional
            If True, only the last `block_size` entries are used.
            If False, all logged trials are used (simple conjugate update with all data).
        """
        d_idx = self._get_distance_index(distance)
        bin_history = self.time_log[distance]

        if len(bin_history) == 0:
            # Nothing to update
            return

        if use_last_block_only:
            if len(bin_history) < block_size:
                # Not enough trials yet; you *could* either skip or use what's available.
                # Here we use whatever we have.
                relevant_bins = bin_history
            else:
                relevant_bins = bin_history[-block_size:]
        else:
            relevant_bins = bin_history

        # Count occurrences per time bin for this distance
        counts = np.zeros(self.num_time_bins, dtype=float)
        for idx in relevant_bins:
            counts[idx] += 1.0

        # Update Dirichlet parameters:
        # dirichlet_alpha[time_bin, distance_idx] += counts[time_bin]
        self.dirichlet_alpha[:, d_idx] += counts

        # Recompute categorical probabilities for this distance:
        alpha_col = self.dirichlet_alpha[:, d_idx]          # shape (num_time_bins,)
        probs_time_given_d = alpha_col / alpha_col.sum()    # normalized

        # Store into categorical_probs row for this distance
        self.categorical_probs[d_idx, :] = probs_time_given_d

    # ---------- Convenience accessors ----------

    def get_time_bin_probabilities(self, distance: float) -> np.ndarray:
        """
        Get current categorical P(time_bin | distance) for a given distance.
        """
        d_idx = self._get_distance_index(distance)
        return self.categorical_probs[d_idx, :].copy()

    def get_dirichlet_parameters(self, distance: float) -> np.ndarray:
        """
        Get current Dirichlet alpha vector over time bins for a given distance.
        Shape: (num_time_bins,)
        """
        d_idx = self._get_distance_index(distance)
        return self.dirichlet_alpha[:, d_idx].copy()


# Example usage / quick sanity check
if __name__ == "__main__":
    patient = Patient()

    d0 = patient.distance_levels[0]

    # Simulate 7 hits at distance d0 with different times
    hit_times = [2.1, 2.4, 4.8, 5.0, 5.1, 6.7, 6.9]
    for t in hit_times:
        patient.map_time_to_bin(d0, t)

    # Update after 5 most recent trials at distance d0
    patient.bayesian_update(d0, block_size=5, use_last_block_only=True)

    print("Time bins:", patient.time_bins)
    print("Dirichlet alpha (first distance):", patient.get_dirichlet_parameters(d0))
    print("Categorical P(time_bin | d0):", patient.get_time_bin_probabilities(d0))
