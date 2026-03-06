"""
compare_qlearning.py
--------------------
Runs Q-learning 1-var, 2-var and 3-var for 5000 trials across all five
canonical patient profiles and saves two figures:

  Assets/qlearning_comparison/rolling_hit_rate.png   — rows=profiles, cols=var-levels
  Assets/qlearning_comparison/caterpillar.png        — mean d (and mean t) per profile × var-level

Run from the repo root:
    python compare_qlearning.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "Assets" / "qlearning_comparison"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod   # must be registered before exec so dataclasses can resolve __module__
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
profiles_mod = _load("patient_profiles_shared", BASE_DIR / "patient_profiles_shared.py")
PATIENT_PROFILES = profiles_mod.PATIENT_PROFILES

patient_mod = _load(
    "patients.patient_simulation_v4",
    BASE_DIR / "patients" / "patient_simulation_v4.py",
)
PatientModel = patient_mod.PatientModel

viz_mod = _load("tests.visualization", BASE_DIR / "tests" / "visualization.py")

ql1 = _load("Qlearning_1var", BASE_DIR / "1_var" / "Qlearning_1var.py")
ql2 = _load("Qlearning_2var", BASE_DIR / "2_var" / "Qlearning_2var.py")
ql3 = _load("Qlearning_3var", BASE_DIR / "3_var" / "Qlearning.py")

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
N_TRIALS = 5000
SEED = 7
T_FIXED_1VAR = 4.0   # fixed time used for 1-var (2-var and 3-var adapt t freely)

PROFILE_NAMES = list(PATIENT_PROFILES.keys())
VAR_LEVELS = ["1-var", "2-var", "3-var"]

VAR_COLORS = {
    "1-var": "#e41a1c",   # red
    "2-var": "#377eb8",   # blue
    "3-var": "#4daf4a",   # green
}

# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
print("Running Q-learning comparison — 5000 trials per (profile × var-level)")
results: dict[str, dict[str, dict]] = {v: {} for v in VAR_LEVELS}

for profile_name, params in PATIENT_PROFILES.items():
    print(f"  {profile_name}")

    # 1-var
    patient = PatientModel(**{**params, "seed": SEED})
    cfg1 = ql1.QLearningConfig()
    logs1, counts1, _ = ql1.run_sim(
        patient=patient, n_trials=N_TRIALS, seed=SEED,
        cfg=cfg1, calibration=True, t_fixed=T_FIXED_1VAR,
    )
    results["1-var"][profile_name] = {"logs": logs1, "counts": counts1}
    hr = np.mean(np.array(logs1["hit"][-500:], dtype=float))
    avg_d = float(np.mean(logs1["d"]))
    print(f"    1-var  hit={hr:.3f}  mean_d={avg_d:.3f}  t_fixed={T_FIXED_1VAR}")

    # 2-var
    patient = PatientModel(**{**params, "seed": SEED})
    cfg2 = ql2.QLearningConfig()
    logs2, counts2, _ = ql2.run_sim(
        patient=patient, n_trials=N_TRIALS, seed=SEED,
        cfg=cfg2, calibration=True,
    )
    results["2-var"][profile_name] = {"logs": logs2, "counts": counts2}
    hr = np.mean(np.array(logs2["hit"][-500:], dtype=float))
    avg_d = float(np.mean(logs2["d"]))
    avg_t = float(np.mean(logs2["t"]))
    print(f"    2-var  hit={hr:.3f}  mean_d={avg_d:.3f}  mean_t={avg_t:.3f}")

    # 3-var
    patient = PatientModel(**{**params, "seed": SEED})
    cfg3 = ql3.QLearningConfig()
    logs3, counts3, _ = ql3.run_sim(
        patient=patient, n_trials=N_TRIALS, seed=SEED,
        cfg=cfg3, calibration=True,
    )
    results["3-var"][profile_name] = {"logs": logs3, "counts": counts3}
    hr = np.mean(np.array(logs3["hit"][-500:], dtype=float))
    avg_d = float(np.mean(logs3["d"]))
    avg_t = float(np.mean(logs3["t"]))
    print(f"    3-var  hit={hr:.3f}  mean_d={avg_d:.3f}  mean_t={avg_t:.3f}")

# ---------------------------------------------------------------------------
# Plot 1: Rolling hit rate matrix
#   rows = patient profiles, cols = var-levels
#   One colored line per var-level
# ---------------------------------------------------------------------------
WINDOW = 100
n_profiles = len(PROFILE_NAMES)
n_vars = len(VAR_LEVELS)

fig, axes = plt.subplots(
    n_profiles, n_vars,
    figsize=(4.5 * n_vars, 2.8 * n_profiles),
    sharex=True, sharey=True, squeeze=False,
)

for row, profile_name in enumerate(PROFILE_NAMES):
    for col, var_level in enumerate(VAR_LEVELS):
        ax = axes[row, col]
        logs = results[var_level][profile_name]["logs"]
        rolling = viz_mod.rolling_hitting_rate({"hit": logs["hit"]}, window=WINDOW, min_periods=1)
        ax.plot(rolling, linewidth=1.5, color=VAR_COLORS[var_level])
        ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="p*=0.7")
        for vx in (500, 1000, 2000, 3000):
            ax.axvline(x=vx, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.set_title(var_level, fontsize=11, color=VAR_COLORS[var_level], fontweight="bold")
        if col == 0:
            ax.set_ylabel(profile_name, fontsize=9)
        if row == n_profiles - 1:
            ax.set_xlabel("Trial", fontsize=9)

fig.suptitle(
    f"Q-learning Rolling Hit Rate (window={WINDOW}) — Profiles × Var-level",
    fontsize=13,
)
fig.tight_layout()
save_path = ASSETS_DIR / "rolling_hit_rate.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {save_path}")

# ---------------------------------------------------------------------------
# Plot 2: Caterpillar — mean distance ± SD  (left panel)
#                       mean time ± SD       (right panel, 2-var and 3-var only)
#   Y-axis = patient profiles (one dot per profile)
#   Three horizontal error bars per cell, one per var-level (colored)
# ---------------------------------------------------------------------------
y_prof = np.arange(n_profiles)

# Offset dots vertically so 3 var-level marks don't overlap
var_offsets = {v: (i - 1) * 0.18 for i, v in enumerate(VAR_LEVELS)}

fig2, (ax_d, ax_t) = plt.subplots(
    1, 2, figsize=(12, 5), sharey=True,
)

# -- Distance panel --
ax_d.set_title("Mean Distance ± SD (m)", fontsize=11)
ax_d.set_xlabel("Distance (m)", fontsize=10)
ax_d.set_ylabel("Patient Profile", fontsize=10)
ax_d.set_yticks(y_prof)
ax_d.set_yticklabels(PROFILE_NAMES, fontsize=9)
ax_d.grid(True, axis="x", alpha=0.3)

for var_level in VAR_LEVELS:
    color = VAR_COLORS[var_level]
    offset = var_offsets[var_level]
    for pi, profile_name in enumerate(PROFILE_NAMES):
        logs = results[var_level][profile_name]["logs"]
        avg_d, sd_d = viz_mod.average_distance(logs)
        ax_d.errorbar(
            avg_d, y_prof[pi] + offset, xerr=sd_d,
            fmt="o", capsize=4, linewidth=1.5,
            color=color, label=var_level if pi == 0 else "_nolegend_",
        )

ax_d.legend(fontsize=9, loc="lower right")

# -- Time panel (2-var and 3-var adapt t; 1-var is constant at T_FIXED_1VAR) --
ax_t.set_title("Mean Time ± SD (s)", fontsize=11)
ax_t.set_xlabel("Time (s)", fontsize=10)
ax_t.grid(True, axis="x", alpha=0.3)

for var_level in VAR_LEVELS:
    color = VAR_COLORS[var_level]
    offset = var_offsets[var_level]
    for pi, profile_name in enumerate(PROFILE_NAMES):
        logs = results[var_level][profile_name]["logs"]
        avg_t, sd_t = viz_mod.average_time(logs)
        ax_t.errorbar(
            avg_t, y_prof[pi] + offset, xerr=sd_t,
            fmt="s", capsize=4, linewidth=1.5,
            color=color, label=var_level if pi == 0 else "_nolegend_",
            # dashed line for 1-var since t is fixed (SD≈0)
            linestyle="--" if var_level == "1-var" else "-",
            alpha=0.6 if var_level == "1-var" else 1.0,
        )

ax_t.legend(fontsize=9, loc="lower right")

fig2.suptitle(
    "Q-learning Caterpillar: Mean ± SD across Patient Profiles",
    fontsize=13,
)
fig2.tight_layout()
save_path2 = ASSETS_DIR / "caterpillar.png"
fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {save_path2}")
