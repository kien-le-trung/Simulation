# patient_simulation.py
import numpy as np


class PatientModel:
    """
    Natural motor model (single mechanism, no "time-mode vs distance-mode" switching).

    Keeps the SAME inputs/outputs your OR code expects:
      sample_trial(t_sys=..., d_sys=..., distance_level=..., previous_hit=...)

    Returns:
      {
        "t_pat": float,
        "d_pat": float,
        "hit": bool,
        "time_ratio": float,
        "dist_ratio": float
      }

    Conventions (per your request):
      - If HIT:  time_ratio = t_pat / t_sys     (<= 1),  dist_ratio = 1
      - If MISS: time_ratio = 1                (NOT 0), dist_ratio = d_pat / d_sys
    """

    def __init__(
        self,
        t_levels=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        k_d_per_sec=0.25,
        sigma_t=0.15,
        sigma_d=0.02,
        seed=7,
        # New model knobs (do NOT change I/O; only simulation behavior)
        v_mean=0.12,          # m/s  (as you requested)
        v_sigma=0.03,         # m/s  (tunable; must keep v mostly > 0)
        r_base=0.25,          # seconds (reaction/latency baseline)
        r_sigma=0.06          # seconds
    ):
        # Keep the structure/attributes similar to the old model
        self.t_levels = np.array(t_levels, dtype=float)
        self.k_d_per_sec = float(k_d_per_sec)
        self.sigma_t = float(sigma_t)
        self.sigma_d = float(sigma_d)

        self.rng = np.random.default_rng(seed)

        # New model params
        self.v_mean = float(v_mean)
        self.v_sigma = float(v_sigma)
        self.r_base = float(r_base)
        self.r_sigma = float(r_sigma)

        # (optional) derived means kept for compatibility / inspection
        self.t_means = self.t_levels.copy()
        self.d_means = self.k_d_per_sec * self.t_means

    def _clip_level(self, distance_level: int) -> int:
        """Distance_level is expected in {0,1,2} from your OR code; clip just in case."""
        return int(np.clip(distance_level, 0, 2))

    def sample_trial(self, *, t_sys: float, d_sys: float, distance_level: int, previous_hit: bool = True):
        """
        One-trial simulation.

        Mechanism:
          - Sample reaction time r (>=0)
          - Sample speed v ~ Normal(mean=0.10, sd=v_sigma), truncated to v>0
          - Time needed to reach target: t_need = r + d_sys / v
            * If t_need <= t_sys => HIT (t_pat=t_need, d_pat=d_sys)
            * Else => MISS (t_pat=t_sys, d_pat=v*max(t_sys-r,0))
        """
        t_sys = float(t_sys)
        d_sys = float(d_sys)
        lvl = self._clip_level(distance_level)

        # ---- 1) sample reaction time (latency) ----
        # Make reaction slightly worse at higher distance_level (still "natural", no rule switching).
        # previous_hit only nudges slightly (confidence/fatigue), NOT a hard mode.
        prev_shift = -0.01 if previous_hit else +0.01
        r_mean = self.r_base + 0.05 * lvl + prev_shift
        r = float(self.rng.normal(loc=r_mean, scale=self.r_sigma))
        r = max(r, 0.0)

        # ---- 2) sample speed v ~ Normal(mean=0.10, sd=...) ----
        v = float(self.rng.normal(loc=self.v_mean, scale=self.v_sigma))
        # Truncate to strictly positive to avoid division issues
        if v <= 1e-6:
            v = 1e-6

        # ---- 3) determine outcome ----
        t_need = r + (d_sys / v)

        if t_need <= t_sys:
            hit = True
            t_pat = t_need
            d_pat = d_sys
            time_ratio = t_pat / max(t_sys, 1e-9)
            dist_ratio = 1.0
        else:
            hit = False
            # distance achieved within available movement time (after reaction)
            move_time = max(t_sys - r, 0.0)
            d_pat = min(v * move_time, d_sys)
            t_pat = t_sys

            # per your request: time_ratio should be 1 on misses (not 0)
            time_ratio = 1.0
            dist_ratio = d_pat / max(d_sys, 1e-9)

        return {
            "t_pat": float(t_pat),
            "d_pat": float(d_pat),
            "hit": bool(hit),
            "time_ratio": float(time_ratio),
            "dist_ratio": float(dist_ratio),
        }