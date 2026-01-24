import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from patient_simulation import PatientModel

class RatioController:
    """
    - Time control: PI on e_t = 1 - time_ratio.
      Learning rate Kp_t is the time_ratio itself (with tiny floor to avoid zero on misses).
      Ki_t = beta_t * Kp_t (beta_t ~ 0.1 by default).
    - Distance control: multiplicative P on e_r = 1 - dist_ratio.
      Learning rate Kp_d is the dist_ratio itself (with floor & cap for stability).
    """
    def __init__(self,
                 beta_t=0.1,              # relates Ki_t to Kp_t (Ki_t = beta_t * Kp_t)
                 min_lr_t=0.05,           # floor for time learning rate (avoid stalling when time_ratio=0)
                 min_lr_d=0.05,           # floor for distance learning rate
                 max_lr_d=1.5,            # cap distance learning rate (dist_ratio can be >1)
                 t_bounds=(1.0, 8.0),
                 d_bounds=(0.2, 2.5)):
        self.beta_t = float(beta_t)
        self.min_lr_t = float(min_lr_t)
        self.min_lr_d = float(min_lr_d)
        self.max_lr_d = float(max_lr_d)
        self.t_min, self.t_max = t_bounds
        self.d_min, self.d_max = d_bounds
        self.int_t = 0.0
        self.int_d = 0.0

    def reset(self):
        self.int_t = 0.0

    def update(self, patient: "PatientModel", t_sys, d_sys, outcome):
        # --- Time PI with adaptive gains ---
        e_t = t_sys - outcome["t_pat"]  # Corrected: positive when patient is too fast
        Kp_t = max(outcome["time_ratio"], self.min_lr_t) # adaptive proportional gain
        Ki_t = self.beta_t * Kp_t                         # adaptive integral gain tied to Kp_t
        self.int_t += e_t
        t_next = t_sys - Kp_t * e_t - Ki_t * self.int_t  # Subtract to decrease time when patient is too fast
        t_next = float(np.clip(t_next, self.t_min, self.t_max))

        # --- Distance multiplicative P with adaptive gain ---
        e_r = outcome["d_pat"] - d_sys  # Corrected: positive when patient overshoots
        # learning rate equals the ratio itself (bounded for stability)
        Kp_d = np.clip(outcome["dist_ratio"], self.min_lr_d, self.max_lr_d)
        Ki_d = self.beta_t * Kp_d
        self.int_d += e_r
        d_next = d_sys + Kp_d * e_r + Ki_d * self.int_d  # Add to increase distance when patient overshoots
        d_next = float(np.clip(d_next, self.d_min, self.d_max))

        return t_next, d_next