# 1_var Mechanisms

Each 1-variable algorithm keeps `t` fixed (default `t_fixed=4.0`) and only adapts `d`.
All trial execution remains through `patients/patient_simulation_v4.py` via the original 2-var implementations.

- `staircasing_1var.py`:
  Uses the original staircase hit/miss streak logic, but forces axis selection to distance only (`_pick_axis -> "d"`).

- `QUEST_1var.py`:
  Uses the same Bayesian posterior update and expected-entropy selection, but constrains stimulus set to `t_grid=[t_fixed]`.

- `logistic_online_1var_v2.py`:
  Keeps online logistic model + target-hit selection logic, but constrains candidate pool and calibration bounds to a single fixed time.

- `operations_research_1var.py`:
  Keeps score-based candidate optimization (effectiveness + variability + ROM penalty), with the search lattice reduced to distance over one fixed time slice.

- `Qlearning_1var.py`:
  Keeps Q-learning state/reward updates, but forces `n_t_bins=1` and fixed time bounds so actions only vary in distance.

- `control_system_1var.py`:
  Keeps margin PI control loop, but forces calibration/expansion time bounds to fixed time so controller outputs only distance changes.
