# patient_simulation.py
import numpy as np
import matplotlib.pyplot as plt


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
        # TIME / DISTANCE MODEL
        t_levels=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        k_d0_per_sec=0.08,
        k_d_decay=0.05,
        sigma_t=0.15,
        sigma_d=0.02,
        seed=7,
        # VELOCITY MODEL
        v_sigma0=0.03,        # m/s  (base sigma; must keep v mostly > 0)
        v_sigma_growth=0.02,   # 1/s  (sigma increases with t_sys)
        # REACTION TIME MODEL
        r_base=0.25,          # seconds (reaction/latency baseline)
        r_sigma=0.06          # seconds
    ):
        # Keep the structure/attributes similar to the old model
        self.t_levels = np.array(t_levels, dtype=float)
        self.k_d0_per_sec = float(k_d0_per_sec)
        self.k_d_decay = float(k_d_decay)
        self.sigma_t = float(sigma_t)
        self.sigma_d = float(sigma_d)

        self.rng = np.random.default_rng(seed)

        # New model params
        self.v_sigma0 = float(v_sigma0)
        self.v_sigma_growth = float(v_sigma_growth)
        self.r_base = float(r_base)
        self.r_sigma = float(r_sigma)

        # (optional) derived means kept for compatibility / inspection
        # If d_levels is provided, it decouples reach distances from t_levels.
        self.t_means = self.t_levels.copy()
        k_t = self.k_d0_per_sec * np.exp(-self.k_d_decay * self.t_means)
        self.d_means = k_t * self.t_means

    def _clip_level(self, distance_level: int) -> int:
        """Distance_level is expected in {0,1,2} from your OR code; clip just in case."""
        return int(np.clip(distance_level, 0, 2))

    def _mean_speed(self, t_sys: float) -> float:
        """Mean speed varies with time via k_d0_per_sec and k_d_decay."""
        return self.k_d0_per_sec * np.exp(-self.k_d_decay * t_sys)

    def _speed_sigma(self, t_sys: float) -> float:
        """Speed variability increases with time via a simple exponential."""
        return self.v_sigma0 * np.exp(self.v_sigma_growth * t_sys)

    def sample_trial(self, *, t_sys: float, d_sys: float, distance_level: int, previous_hit: bool = True):
        """
        One-trial simulation.

        Mechanism:
          - Sample reaction time r (>=0)
          - Sample speed v ~ Normal(mean=k(t), sd=v_sigma(t)), truncated to v>0
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
        v_mean = self._mean_speed(t_sys)
        v_sigma = self._speed_sigma(t_sys)
        v = float(self.rng.normal(loc=v_mean, scale=v_sigma))
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

    def plot_v_t(
        self,
        t_min=None,
        t_max=None,
        d_min=0.0,
        d_max=1.0,
        t_steps=160,
        d_steps=160,
        n_samples=200,
        seed=0,
        stat="mean",
    ):
        import matplotlib.pyplot as plt

        if t_min is None:
            t_min = float(np.min(self.t_levels))
        if t_max is None:
            t_max = float(np.max(self.t_levels))

        t_grid = np.linspace(t_min, t_max, int(t_steps))
        v_mean = self._mean_speed(t_grid)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_grid, v_mean, linewidth=2.0)
        ax.set_title("v_mean vs t_sys")
        ax.set_xlabel("t_sys")
        ax.set_ylabel("v_mean (m/s)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig, ax