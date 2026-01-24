import numpy as np

class PatientModel:
    """
    Matched-difficulty PAIRS of (d, t) distributions.
    Sampling per your spec:
      1) Map d_sys -> closest bucket with d_mean <= d_sys
      2) Sample d_pat ~ N(d_mean, sigma_d) and t_pat ~ N(t_mean, sigma_t)
      3) hit := (t_pat <= t_sys)
      4) If hit: time_ratio = t_pat/t_sys (<=1), dist_ratio = 1
      5) If miss: time_ratio = 0, dist_ratio = d_mean/d_sys
    """
    def __init__(self,
                 t_levels=(1., 2., 3., 4., 5., 6., 7., 8.),
                 k_d_per_sec=0.1,
                 sigma_t=0.15,
                 sigma_d=0.05,
                 seed=7):
        self.rng = np.random.default_rng(seed)
        self.t_means = np.asarray(t_levels, dtype=float)
        self.d_means = k_d_per_sec * self.t_means
        self.sigma_t = float(sigma_t)
        self.sigma_d = float(sigma_d)

    # def _bucket_by_distance(self, d_sys):
    #     """
    #     Find the closest d_mean that is <= d_sys.
    #     If none, fallback to the smallest bucket (index 0).
    #     """
    #     candidates = np.where(self.d_means <= d_sys)[0]
    #     if len(candidates) == 0:
    #         return 0
    #     return int(candidates[-1])

    def sample_trial(self, t_sys, d_sys, distance_level, previous_hit=True):
        """
        Sample a trial with mode switching based on previous outcome.

        Parameters
        ----------
        t_sys : float
            System target time
        d_sys : float
            System target distance
        distance_level : int
            Distance level index
        previous_hit : bool
            Whether the previous trial was a hit. Determines success mode:
            - True: use time-based success (hit if t_pat <= t_sys)
            - False: use distance-based success (hit if d_pat >= d_sys)
        """
        d_mean = self.d_means[distance_level]
        t_mean = self.t_means[distance_level]

        # Sample time from patient's natural distribution
        t_pat = self.rng.normal(loc=t_mean, scale=self.sigma_t)
        t_pat = max(0.02, t_pat)

        if previous_hit:
            # TIME-BASED SUCCESS MODE (after a hit)
            # Hit criterion: finish in time
            if t_pat <= t_sys:
                # HIT: Patient finishes in time → reaches target distance
                d_pat = d_sys
                d_pat = max(0.0, d_pat)
                hit = True
                time_ratio = t_pat / max(t_sys, 1e-9)
                dist_ratio = 1.0
            else:
                # MISS: Patient runs out of time → falls short
                time_used_ratio = t_sys / t_pat
                d_pat = d_mean * time_used_ratio
                d_pat = self.rng.normal(loc=d_pat, scale=self.sigma_d)
                d_pat = max(0.0, d_pat)
                hit = False
                time_ratio = 0.0  # No time adjustment on miss
                dist_ratio = d_pat / max(d_sys, 1e-9)
        else:
            # DISTANCE-BASED SUCCESS MODE (after a miss)
            # Sample distance from patient's natural distribution
            d_pat = self.rng.normal(loc=d_mean, scale=self.sigma_d)
            d_pat = max(0.0, d_pat)

            # Hit criterion: reach the distance
            if d_pat >= d_sys:
                # HIT: Patient reaches distance
                hit = True
                time_ratio = t_pat / max(t_sys, 1e-9)
                dist_ratio = d_pat / max(d_sys, 1e-9)
            else:
                # MISS: Patient falls short
                hit = False
                time_ratio = 0
                dist_ratio = d_pat / max(d_sys, 1e-9)

        return {
            "t_pat": t_pat,
            "d_pat": d_pat,
            "hit": hit,
            "time_ratio": time_ratio,
            "dist_ratio": dist_ratio
        }