import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Reward -> Grade mapping
# -----------------------------

def calculate_reward(outcome, hit_time_thresholds=None, miss_dist_thresholds=None):
    """
    Calculate 5-level graded reward based on time_ratio and dist_ratio.

    Returns reward in {0.0, 0.25, 0.5, 0.75, 1.0}
    """
    if hit_time_thresholds is None:
        hit_time_thresholds = [0.8, 0.6, 0.4, 0.2]
    if miss_dist_thresholds is None:
        miss_dist_thresholds = [0.8, 0.6, 0.4, 0.2]

    reward_levels = [1.0, 0.75, 0.5, 0.25, 0.0]

    hit = outcome["hit"]
    time_ratio = outcome["time_ratio"]
    dist_ratio = outcome["dist_ratio"]

    if hit:
        for i, threshold in enumerate(hit_time_thresholds):
            if time_ratio >= threshold:
                return reward_levels[i]
        return reward_levels[-1]
    else:
        for i, threshold in enumerate(miss_dist_thresholds):
            if dist_ratio >= threshold:
                return reward_levels[i]
        return reward_levels[-1]


def reward_to_grade(reward: float) -> str:
    """
    Map reward {1.0,0.75,0.5,0.25,0.0} -> grade.
    """
    # Be robust to float representation issues.
    if reward >= 0.875:
        return "excellent"
    if reward >= 0.625:
        return "good"
    if reward >= 0.375:
        return "moderate"
    if reward >= 0.125:
        return "bad"
    return "horrible"


# -----------------------------
# Virtual-trial Beta updates
# -----------------------------

GRADE_UPDATES = {
    "excellent": (1.0, 0.0),  # +2 alpha
    "good":      (1.0, 0.0),  # +1 alpha
    "moderate":  (0.0, 0.0),  # no change
    "bad":       (0.0, 1.0),  # +1 beta
    "horrible":  (0.0, 1.0),  # +2 beta
}


@dataclass
class AngleBins:
    n_az: int = 4
    n_el: int = 2
    # Safe "cone" in front of patient:
    # azimuth: left/right (radians), elevation: up/down (radians)
    az_min: float = -math.radians(40.0)
    az_max: float =  math.radians(40.0)
    el_min: float = -math.radians(20.0)
    el_max: float =  math.radians(20.0)


class SpatialThompsonSampler:
    """
    Thompson sampling over (azimuth,elevation) bins with Beta posteriors.
    Uses virtual-trial updates based on 5-level grades.
    """
    def __init__(self, bins: AngleBins, alpha0: float = 1.0, beta0: float = 1.0, seed: Optional[int] = None):
        self.bins = bins
        self.alpha = [[alpha0 for _ in range(bins.n_el)] for _ in range(bins.n_az)]
        self.beta  = [[beta0  for _ in range(bins.n_el)] for _ in range(bins.n_az)]
        self.rng = random.Random(seed)

    def _sample_beta(self, a: float, b: float) -> float:
        # random.Random has betavariate
        return self.rng.betavariate(a, b)

    def select_bin(self) -> Tuple[int, int]:
        """
        Standard TS: sample p_hat for each bin and pick the closest to 0.6.
        """
        best = None
        best_val = 5.0  # large
        for i in range(self.bins.n_az):
            for j in range(self.bins.n_el):
                val = self._sample_beta(self.alpha[i][j], self.beta[i][j])
                val = abs(val - 0.6)
                if val < best_val:
                    best_val = val
                    best = (i, j)
        return best  # type: ignore

    def bin_to_angles(self, i: int, j: int) -> Tuple[float, float]:
        """
        Sample a (theta, phi) uniformly inside bin (i,j).
        theta = azimuth, phi = elevation
        """
        az_w = (self.bins.az_max - self.bins.az_min) / self.bins.n_az
        el_w = (self.bins.el_max - self.bins.el_min) / self.bins.n_el

        az_lo = self.bins.az_min + i * az_w
        az_hi = az_lo + az_w

        el_lo = self.bins.el_min + j * el_w
        el_hi = el_lo + el_w

        theta = self.rng.uniform(az_lo, az_hi)
        phi   = self.rng.uniform(el_lo, el_hi)
        return theta, phi

    def update_from_outcome(self, i: int, j: int, outcome: Dict) -> str:
        """
        Update (alpha,beta) for bin based on graded reward.
        Returns the grade string.
        """
        reward = calculate_reward(outcome)
        grade = reward_to_grade(reward)
        da, db = GRADE_UPDATES[grade]
        self.alpha[i][j] += da
        self.beta[i][j]  += db
        return grade


# -----------------------------
# Geometry: distance d -> (x,y,z)
# -----------------------------

def angles_to_direction(theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert azimuth theta and elevation phi to a unit direction vector.
    Convention:
      x: right
      y: up
      z: forward
    """
    # Using: x = cos(theta)cos(phi), y = sin(phi), z = sin(theta)cos(phi)
    x = math.cos(theta) * math.cos(phi)
    y = math.sin(phi)
    z = math.sin(theta) * math.cos(phi)

    # Normalize (should be ~1 already)
    norm = math.sqrt(x*x + y*y + z*z)
    if norm == 0:
        return (0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm)


def direction_to_point(d: float, direction: Tuple[float, float, float],
                       origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                       y_clamp: Optional[Tuple[float, float]] = (-0.2, 0.35)) -> Tuple[float, float, float]:
    """
    Convert unit direction + distance to a point; optionally clamp y for safety.
    If clamping y changes length, re-project onto the sphere approximately.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction

    x = ox + d * dx
    y = oy + d * dy
    z = oz + d * dz

    if y_clamp is not None:
        y_min, y_max = y_clamp
        y_clamped = max(y_min, min(y, y_max))
        if y_clamped != y:
            # Reproject to keep radius ~d while holding y fixed.
            # Compute remaining radius in xz plane:
            y = y_clamped
            ry = y - oy
            rem2 = max(d*d - ry*ry, 1e-12)
            rem = math.sqrt(rem2)

            # Keep original xz direction and scale to 'rem'
            vx = x - ox
            vz = z - oz
            v_norm = math.sqrt(vx*vx + vz*vz)
            if v_norm < 1e-12:
                # fallback: forward
                vx, vz, v_norm = 0.0, 1.0, 1.0
            x = ox + rem * (vx / v_norm)
            z = oz + rem * (vz / v_norm)

    return (x, y, z)


# -----------------------------
# Main entrypoint
# -----------------------------

# Global sampler so it "remembers" across calls.
_GLOBAL_SAMPLER: Optional[SpatialThompsonSampler] = None


def spatial_decomposition(d: float,
                          outcome: Optional[Dict] = None,
                          origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                          bins: Optional[AngleBins] = None,
                          seed: Optional[int] = 123) -> Tuple[float, float, float]:
    """
    Main function.

    Parameters
    ----------
    d : float
        Distance in meters. Must be in [0.1, 0.8].
    outcome : dict or None
        If provided, uses it to update the *previously selected* bin.
        Expected keys: hit (bool), time_ratio (float), dist_ratio (float).
        - time_ratio: used when hit=True (higher => more challenging attempt)
        - dist_ratio: used when hit=False (higher => closer attempt)
    origin : (float,float,float)
        Coordinate origin (player position). Default (0,0,0).
    bins : AngleBins or None
        Bin settings. If None, uses defaults.
    seed : int or None
        Seed for reproducibility of direction sampling.

    Returns
    -------
    (x,y,z) : tuple of float
        Spawn point coordinates at distance d from origin.

    Notes
    -----
    This function keeps a global Thompson sampler so successive calls adapt.
    Workflow in a real game loop:
      1) call spatial_decomposition(d) -> spawn (x,y,z)
      2) run trial, compute outcome dict
      3) call spatial_decomposition(d, outcome=outcome) to update and get next (x,y,z)
    """
    if not (0.1 <= d <= 0.8):
        raise ValueError(f"d must be between 0.1 and 0.8 meters, got {d}")

    global _GLOBAL_SAMPLER
    if _GLOBAL_SAMPLER is None:
        _GLOBAL_SAMPLER = SpatialThompsonSampler(bins=bins or AngleBins(), seed=seed)

    sampler = _GLOBAL_SAMPLER

    # We need to remember which bin generated the previous point for updating.
    # We'll store it as attributes on the sampler (simple, no external state).
    if not hasattr(sampler, "_last_bin"):
        sampler._last_bin = None  # type: ignore

    # If an outcome is given, update the posterior for the bin that produced the last spawn.
    if outcome is not None and sampler._last_bin is not None:
        i_last, j_last = sampler._last_bin
        sampler.update_from_outcome(i_last, j_last, outcome)

    # Select new bin and sample angles within it
    i, j = sampler.select_bin()
    sampler._last_bin = (i, j)

    theta, phi = sampler.bin_to_angles(i, j)
    direction = angles_to_direction(theta, phi)
    point = direction_to_point(d, direction, origin=origin, y_clamp=(-0.2, 0.35))
    return point


# -----------------------------
# Example usage (remove in production)
# -----------------------------
if __name__ == "__main__":
    # First spawn
    p1 = spatial_decomposition(0.5)
    print("spawn1:", p1)

    # Suppose patient missed but got close (dist_ratio high)
    outcome1 = {"hit": False, "time_ratio": 0.0, "dist_ratio": 0.7}
    p2 = spatial_decomposition(0.5, outcome=outcome1)
    print("spawn2:", p2)

    # Suppose next trial is a hit with high time_ratio
    outcome2 = {"hit": True, "time_ratio": 0.85, "dist_ratio": 0.0}
    p3 = spatial_decomposition(0.5, outcome=outcome2)
    print("spawn3:", p3)
