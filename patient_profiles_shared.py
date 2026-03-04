from __future__ import annotations

# Canonical patient profiles used across 0_var/1_var/2_var/3_var control panels.
PATIENT_PROFILES = {
    "overall_weak": {
        "k_d0_per_sec": 0.15,
        "k_d_decay": 1.0,
        "v_sigma0": 0.04,
        "v_sigma_growth": 0.04,
        "max_reach": 0.40,
        "handedness": 0.8,
        "k_dir_decay": 2.0,
    },
    "overall_medium": {
        "k_d0_per_sec": 0.7,
        "k_d_decay": 0.55,
        "v_sigma0": 0.10,
        "v_sigma_growth": 0.024,
        "max_reach": 0.70,
        "handedness": 0.2,
        "k_dir_decay": 0.5,
    },
    "overall_strong": {
        "k_d0_per_sec": 2.0,
        "k_d_decay": 0.1,
        "v_sigma0": 0.20,
        "v_sigma_growth": 0.008,
        "max_reach": 1.0,
        "handedness": 0.0,
        "k_dir_decay": 0.1,
    },
    "highspeed_lowrom": {
        "k_d0_per_sec": 1.5,
        "k_d_decay": 0.1,
        "v_sigma0": 0.12,
        "v_sigma_growth": 0.01,
        "max_reach": 0.50,
        "handedness": 0.8,
        "k_dir_decay": 2.0,
    },
    "lowspeed_highrom": {
        "k_d0_per_sec": 0.2,
        "k_d_decay": 0.1,
        "v_sigma0": 0.06,
        "v_sigma_growth": 0.04,
        "max_reach": 1.0,
        "handedness": 0.0,
        "k_dir_decay": 0.1,
    },
}

