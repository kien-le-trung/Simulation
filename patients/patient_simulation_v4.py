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
        d_levels=np.linspace(0.1, 1.0, 10),
        k_d0_per_sec=0.15,
        k_d_decay=0,
        sigma_t=0.15,
        sigma_d=0.02,
        seed=7,
        # VELOCITY MODEL
        v_sigma0=0.03,        # m/s  (base sigma; must keep v mostly > 0)
        v_sigma_growth=0.02,   # 1/s  (sigma increases with t_sys)
        # REACTION TIME MODEL
        r_base=0.0,          # seconds (reaction/latency baseline)
        r_sigma=0.0,          # seconds
        # SPATIAL DIRECTION MODEL
        spatial_strength_map=None,
        max_reach_map=None,
    ):
        # Keep the structure/attributes similar to the old model
        self.d_levels = np.array(d_levels, dtype=float)
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

        # 9-bin (3 elevation x 3 azimuth) strength map for front hemisphere.
        # Azimuth bins (left->right): [left, center, right]
        # Elevation bins (low->high): [lower, mid, upper]
        # Values are multiplicative factors on velocity; right hand = harder left.
        # Indexing: idx = elevation * 3 + azimuth
        if spatial_strength_map is None:
            self.spatial_strength_map = np.array(
                [0.50, 0.60, 0.70, 0.75, 0.85, 0.95, 0.90, 0.98, 1.00],
                dtype=float,
            )
        else:
            self.spatial_strength_map = np.array(spatial_strength_map, dtype=float)

        # Per-direction ROM ceiling (meters), indexed by direction idx in [0..8].
        # Default is derived from spatial strength:
        #   max_reach_map = max(d_levels) * (strength / max_strength)
        # so the strongest direction keeps full range and weaker directions get
        # proportionally smaller reachable distance.
        if max_reach_map is None:
            base_max_reach = float(np.max(self.d_levels))
            strength = np.asarray(self.spatial_strength_map, dtype=float)
            strength_max = float(np.max(strength)) if strength.size > 0 else 1.0
            strength_norm = strength / max(strength_max, 1e-9)
            self.max_reach_map = base_max_reach * strength_norm
        else:
            self.max_reach_map = np.asarray(max_reach_map, dtype=float)
        if self.max_reach_map.shape != (9,):
            raise ValueError("max_reach_map must have shape (9,).")
        self.max_reach_map = np.clip(self.max_reach_map, 1e-6, None)
        # Per-direction Beta(1,1) priors for success probability.
        self.spatial_success_alpha = np.ones(9, dtype=float)
        self.spatial_success_beta = np.ones(9, dtype=float)

        # derive time taken to reach each d_level at mean speed
        self.d_means = self.d_levels
        self.t_levels = d_levels / self._mean_speed(self.d_levels)

    def _clip_level(self, distance_level: int) -> int:
        """Distance_level is expected in {0,1,2} from your OR code; clip just in case."""
        return int(np.clip(distance_level, 0, 2))

    def _mean_speed(self, d_sys: float) -> float:
        """Mean speed varies with distance via k_d0_per_sec and k_d_decay."""
        return self.k_d0_per_sec * np.exp(-self.k_d_decay * d_sys)

    def _speed_sigma(self, d_sys: float) -> float:
        """Speed variability increases with time via a simple exponential."""
        return self.v_sigma0 * np.exp(self.v_sigma_growth * d_sys)

    def sample_trial(
        self,
        *,
        t_sys: float,
        d_sys: float,
        distance_level: int,
        previous_hit: bool = True,
        direction_bin: int | None = None,
    ):
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
        lvl = distance_level
        idx = None if direction_bin is None else int(np.clip(direction_bin, 0, 8))
        if idx is None:
            max_reach_dir = float(np.max(self.max_reach_map))
        else:
            max_reach_dir = float(self.max_reach_map[idx])

        # Hard ROM edge-case guard: if requested distance exceeds directional
        # reachable ceiling, immediately return a miss at timeout.
        if d_sys > max_reach_dir:
            t_pat = t_sys
            d_pat = max_reach_dir
            hit = False
            time_ratio = 1.0
            dist_ratio = d_pat / max(d_sys, 1e-9)
            return {
                "t_pat": float(t_pat),
                "d_pat": float(d_pat),
                "hit": bool(hit),
                "time_ratio": float(time_ratio),
                "dist_ratio": float(dist_ratio),
            }

        # ---- 1) sample reaction time (latency) ----
        # Make reaction slightly worse at higher distance_level (still "natural", no rule switching).
        # previous_hit only nudges slightly (confidence/fatigue), NOT a hard mode.
        prev_shift = -0.01 if previous_hit else +0.01
        r_mean = self.r_base + 0.05 * lvl + prev_shift
        r = float(self.rng.normal(loc=r_mean, scale=self.r_sigma))
        r = max(r, 0.0)

        # ---- 2) sample speed v ~ Normal(mean=0.10, sd=...) ----
        v_mean = self._mean_speed(d_sys)
        if idx is not None:
            v_mean *= float(self.spatial_strength_map[idx])
        v_sigma = self._speed_sigma(d_sys)
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

    def calibration(
        self,
        *,
        previous_hit: bool = True,
        direction_bin: int | None = None,
        shuffle: bool = True,
        n_per_direction: int = 3,
        t_cap: float = 10.0,
    ):
        """
        Run directional calibration probes with timeout-censored outcomes.

        Process:
          - For each sampled direction bin, pick `n_per_direction` distance candidates
            that span the available d_levels.
          - For each candidate, run one trial with fixed time cap `t_cap`.
          - A miss at `t_cap` is a censored observation (target not reached in time).

        By default:
          - If `direction_bin is None`, probe all 9 bins.
          - If `direction_bin` is provided, probe only that bin.
        """
        d_levels = np.sort(np.asarray(self.d_levels, dtype=float))
        if d_levels.size == 0:
            raise ValueError("d_levels must be non-empty for calibration.")
        if int(n_per_direction) <= 0:
            raise ValueError("n_per_direction must be a positive integer.")
        if float(t_cap) <= 0.0:
            raise ValueError("t_cap must be positive.")

        if direction_bin is None:
            directions = list(range(9))
        else:
            directions = [int(np.clip(direction_bin, 0, 8))]

        idx = np.round(
            np.linspace(0, d_levels.size - 1, int(n_per_direction))
        ).astype(int)
        d_candidates = [float(d_levels[i]) for i in idx]

        probes = [(direction, d_sys) for direction in directions for d_sys in d_candidates]
        if shuffle:
            order = self.rng.permutation(len(probes))
            probes = [probes[i] for i in order]

        trials = []
        for direction, d_sys in probes:
            distance_level = int(np.argmin(np.abs(self.d_levels - d_sys)))
            outcome = self.sample_trial(
                t_sys=float(t_cap),
                d_sys=d_sys,
                distance_level=distance_level,
                previous_hit=bool(previous_hit),
                direction_bin=direction,
            )
            reached = bool(outcome["hit"])
            t_pat_obs = float(min(outcome["t_pat"], t_cap))
            trials.append(
                {
                    "direction_bin": int(direction),
                    "d_sys": float(d_sys),
                    "t_cap": float(t_cap),
                    "t_sys": float(t_cap),  # compatibility alias
                    "distance_level": distance_level,
                    "t_pat_obs": t_pat_obs,
                    "reached": reached,
                    "censored": bool(not reached),
                    **outcome,
                }
            )

        d_list = [float(x["d_sys"]) for x in trials]
        t_list = [float(x["t_cap"]) for x in trials]
        dir_list = [int(x["direction_bin"]) for x in trials]
        hit_list = [bool(x["hit"]) for x in trials]
        reached_list = [bool(x["reached"]) for x in trials]
        censored_list = [bool(x["censored"]) for x in trials]
        t_pat_obs_list = [float(x["t_pat_obs"]) for x in trials]
        t_ratio_list = [float(x["time_ratio"]) for x in trials]
        d_ratio_list = [float(x["dist_ratio"]) for x in trials]

        hits = np.array(hit_list, dtype=float)
        t_ratio = np.array(t_ratio_list, dtype=float)
        d_ratio = np.array(d_ratio_list, dtype=float)

        per_direction = {}
        for direction in directions:
            rows = [x for x in trials if int(x["direction_bin"]) == int(direction)]
            if len(rows) == 0:
                continue
            reached_arr = np.array([bool(r["reached"]) for r in rows], dtype=float)
            t_obs_arr = np.array([float(r["t_pat_obs"]) for r in rows], dtype=float)
            d_arr = np.array([float(r["d_sys"]) for r in rows], dtype=float)
            per_direction[int(direction)] = {
                "n_probes": int(len(rows)),
                "n_reached": int(np.sum(reached_arr)),
                "n_censored": int(len(rows) - np.sum(reached_arr)),
                "reach_rate": float(np.mean(reached_arr)),
                "mean_t_pat_obs": float(np.mean(t_obs_arr)),
                "mean_d_sys": float(np.mean(d_arr)),
            }

        return {
            "trials": trials,
            "n_trials": len(trials),
            "direction_bins": directions,
            "d_candidates": d_candidates,
            "t_cap": float(t_cap),
            "d": d_list,
            "t": t_list,
            "direction": dir_list,
            "hit": hit_list,
            "reached": reached_list,
            "censored": censored_list,
            "t_pat_obs": t_pat_obs_list,
            "t_ratio": t_ratio_list,
            "d_ratio": d_ratio_list,
            "hit_rate": float(hits.mean()) if len(hits) > 0 else 0.0,
            "mean_time_ratio": float(t_ratio.mean()) if len(t_ratio) > 0 else 0.0,
            "mean_dist_ratio": float(d_ratio.mean()) if len(d_ratio) > 0 else 0.0,
            "per_direction": per_direction,
        }
