# Parkinson Project — Simulation

## Overview

This project simulates adaptive exercise protocols for Parkinson's patients. Each trial presents a reaching task defined by a **distance** `d` (metres) and a **time limit** `t` (seconds). Different adaptive algorithms decide how to adjust `d` and/or `t` trial-by-trial to keep the patient appropriately challenged. Patient behaviour is simulated by a probabilistic motor model.

The project is organised by how many variables the algorithm is allowed to adapt simultaneously: 0, 1, 2, or 3.

---

## Project Structure

```
Simulation/
│
├── patient_profiles_shared.py      # Canonical patient parameter definitions (shared across all var-levels)
│
├── patients/                       # Patient motor models
│   ├── patient_simulation_v4.py    # CANONICAL model (use this one)
│   ├── patient_simulation_v3.py    # Earlier version
│   ├── patient_simulation_v2.py    # Earlier version
│   └── patient_simulation.py      # Original prototype
│
├── tests/                          # Shared evaluation and plotting utilities
│   ├── visualization.py            # All reusable plotting functions (hit rate, caterpillar, heatmaps, etc.)
│   ├── make_ideal_distribution.py  # Computes the theoretical ideal (d,t) trial distribution for a patient
│   └── evaluation.py              # Legacy evaluation script (early prototype)
│
├── 0_var/                          # Baseline: no adaptation (random or fixed parameters)
│   └── algorithm_0_var.py          # Random policy + its own run/plot entry point
│
├── 1_var/                          # Adapts distance only; time is fixed per run
│   ├── simulation_control_panel_1var.py   # ENTRY POINT — runs all algorithms, saves all plots
│   ├── control_system_1var.py             # PI controller on continuous distance margin
│   ├── operations_research_1var.py        # OR-based optimisation
│   ├── staircasing_1var.py                # 2-up/2-down staircase on distance
│   ├── logistic_online_1var.py            # Online logistic regression threshold estimation
│   ├── Qlearning_1var.py                  # Q-learning over discretised distance levels
│   ├── QUEST_1var.py                      # QUEST psychometric threshold estimation
│   └── MECHANISMS.md                      # Notes on algorithm mechanics
│
├── 2_var/                          # Adapts both distance and time
│   ├── simulation_control_panel_2var.py
│   ├── control_system_2var.py
│   ├── operations_research_2var.py
│   ├── staircasing_2var.py
│   ├── logistic_online_2var.py
│   ├── Qlearning_2var.py
│   └── QUEST_2var.py
│
├── 3_var/                          # Adapts distance, time, and direction
│   ├── simulation_control_panel_3var.py
│   ├── control_system_3var.py
│   ├── operations_research_3var.py
│   ├── staircasing_3var.py
│   ├── logistic_online_3var.py
│   ├── Qlearning.py / QUEST_3var.py / hybrid_adaptive_3var.py
│   └── (several prototype/versioned variants)
│
├── Assets/                         # All generated plots (auto-created on run, safe to delete)
│   ├── 0_var/
│   ├── 1_var/
│   ├── 2_var/
│   └── 3_var/
│
└── (root-level legacy files)       # Early prototypes — not part of the active workflow
    ├── simulation.py / simulation_control_panel.py / simulation_control_panel_v2.py
    ├── controller.py / staircasing.py / operations_research.py / patient_class.py
    ├── spatial_decomposition_TS.py / contextual_bandits.py
    ├── analyze_speed_bias.py / diagnose_oscillation.py / investigate_bias.py
    └── *.png / *.csv                   # Scratch outputs from legacy runs
```

---

## Patient Profiles

Defined once in `patient_profiles_shared.py` and imported by every control panel. There are five canonical profiles:

| Profile | Character |
|---|---|
| `overall_weak` | Slow speed, small ROM (max_reach 0.40 m), strong handedness |
| `overall_medium` | Moderate speed and ROM (max_reach 0.70 m) |
| `overall_strong` | Fast speed, full ROM (max_reach 1.00 m) |
| `highspeed_lowrom` | Fast but limited range (max_reach 0.50 m) |
| `lowspeed_highrom` | Full range but slow (max_reach 1.00 m) |

Each profile specifies: `k_d0_per_sec` (mean speed), `k_d_decay` (speed drop-off with distance), `v_sigma0` / `v_sigma_growth` (speed variability), `max_reach`, `handedness`, `k_dir_decay`.

---

## Data Flow During a Trial Run

```
simulation_control_panel_{n}var.py
        │
        │  reads patient params from patient_profiles_shared.py
        │  wraps PatientModel as _FixedTimeRandomDirectionPatient
        │  (enforces t_sys = t_fixed on every trial call)
        │
        ▼
run_algorithm(algorithm_name, patient_profile, n_trials, calibration=False)
        │
        │  dynamically loads the algorithm module
        │  calls run_sim(patient, n_trials, t_fixed, ...)
        │
        ▼
Algorithm run_sim() loop  [e.g. control_system_1var.run_sim]
        │
        │  for each trial k:
        │  ┌─────────────────────────────────────────────────────────┐
        │  │  1. Algorithm proposes (d_sys, t_sys)                   │
        │  │     e.g. d = dmin + u * (dmax - dmin)                   │
        │  │                                                          │
        │  │  2. patient.sample_trial(t_sys, d_sys, ...)             │
        │  │     ├─ samples reaction time r                          │
        │  │     ├─ samples speed v ~ N(v_mean, v_sigma)             │
        │  │     ├─ computes t_need = r + d_sys / v                  │
        │  │     ├─ HIT  if t_need <= t_sys                          │
        │  │     └─ MISS otherwise (ROM guard if d > max_reach)      │
        │  │                                                          │
        │  │  3. Outcome returned:                                    │
        │  │     { hit, t_pat, d_pat, time_ratio, dist_ratio }       │
        │  │                                                          │
        │  │  4. Algorithm updates its internal state                 │
        │  │     e.g. PI controller, staircase streak, Q-table, etc. │
        │  └─────────────────────────────────────────────────────────┘
        │
        │  returns (logs dict, counts matrix)
        │
        ▼
Control panel collects logs for all (t_fixed × profile × algorithm)
        │
        ▼
Plotting functions  →  Assets/{n}_var/*.png
```

### Key data structures

**`logs` dict** (returned by every `run_sim`):

| Key | Description |
|---|---|
| `d` | Distance proposed by the algorithm each trial (metres) |
| `t` | Time limit used each trial (seconds) |
| `hit` | 1 = hit, 0 = miss |
| `dist_ratio` | `d_pat / d_sys` (1.0 on a hit) |
| `time_ratio` | `t_pat / t_sys` on hit; 1.0 on miss |
| `margin` | Algorithm-specific margin signal |
| `m_hat` | EWMA-smoothed margin (control system) |

**`counts`** — 5×5 numpy array counting trials per (distance-bin, time-bin) cell.

---

## How to Run

Each var-level has a single entry point. Run from the repo root:

```bash
# 0-variable baseline (random policy)
python 0_var/algorithm_0_var.py

# 1-variable (distance only, time fixed)
python 1_var/simulation_control_panel_1var.py

# 2-variable (distance + time)
python 2_var/simulation_control_panel_2var.py

# 3-variable (distance + time + direction)
python 3_var/simulation_control_panel_3var.py
```

Each script runs all algorithms against all patient profiles, prints per-algorithm hit rates to stdout, and saves all plots to `Assets/{n}_var/`.

**Configuring the experiment** (edit inside the `if __name__ == "__main__":` block of the relevant control panel):

| Variable | Location | Effect |
|---|---|---|
| `T_FIXED_VALUES` | `simulation_control_panel_1var.py:457` | List of fixed times to sweep (1-var only) |
| `T_COLORS` | same file | Colour per time value in all plots |
| `n_trials` | `run_algorithm(n_trials=...)` call | Trials per run |
| `algorithms` | list in `__main__` | Which algorithms to include |

---

## Modifying Visualizations — What to Touch and What Not To

### Safe to modify (visualizations only)

| File | What you can change |
|---|---|
| `tests/visualization.py` | Any function: `plot_rolling_hit_rate`, `plot_caterpillar_means`, `plot_hit_rate_matrix`, `plot_heatmap`, `rolling_hitting_rate`, `average_time`, `average_distance`, etc. These only read `logs` dicts and produce matplotlib figures. |
| `simulation_control_panel_1var.py` lines 511–707 | All four plotting blocks (hit rate matrix, d-level counts, caterpillar, d×t heatmaps). Change colours, figure sizes, axis labels, layout, save paths, etc. |
| `simulation_control_panel_2var.py` (plotting section) | Same — everything after the experiment loop. |
| `simulation_control_panel_3var.py` (plotting section) | Same. |
| `T_COLORS` dict | Change the colour assigned to each `t_fixed` value. |
| `Assets/` folder | Delete freely — all plots are regenerated on each run. |

### Do NOT modify (simulation logic)

| File | Why it must not be touched |
|---|---|
| `patients/patient_simulation_v4.py` | The canonical patient motor model. Changes here alter the ground truth of every simulation. |
| `patient_profiles_shared.py` | Patient parameters. Changing these changes what patients are being simulated, invalidating comparisons. |
| `run_sim()` in any algorithm file | Core adaptive logic. Modifying these changes the algorithm behaviour, not just how results are displayed. |
| `tests/make_ideal_distribution.py` | Computes the theoretical optimal (d,t) distribution used as a benchmark reference. |
| The **experiment loop** in each control panel | Lines 474–509 in `simulation_control_panel_1var.py` (the `results_by_t` collection loop). This is where data is generated — it must not be changed to ensure results are valid before plotting. |
| `run_algorithm()` function in any control panel | The dispatch layer that constructs the patient wrapper and routes calls to algorithm modules. |

### The clean boundary

The experiment loop returns `results_by_t[t_fixed][profile][algorithm]` containing `logs` and `counts`. Everything **before** this point is simulation; everything **after** is visualization. The safe edit zone starts at the comment:

```python
# Plot 1: Rolling hit rate matrix ...
```

in each control panel's `__main__` block.
