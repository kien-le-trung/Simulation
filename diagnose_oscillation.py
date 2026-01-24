"""
Diagnostic script to understand why the system oscillates between easy and hard tasks.
Analyzes the feedback loops and mode switching behavior.
"""
import numpy as np
import matplotlib.pyplot as plt
from operations_research import run_controller

# Run the controller and capture detailed trace
hist, counts = run_controller(n_trials=100, seed=7)

# Extract data
hits = np.array(hist["hit"])
d_vals = np.array(hist["d"])
t_vals = np.array(hist["t"])
v_hat_vals = np.array(hist["v_hat"])
sigma_v_vals = np.array(hist["sigma_v"])
p_pred_vals = np.array(hist["p_pred"])
v_req_vals = np.array(hist["v_req"])

# Identify the oscillation pattern
print("=" * 70)
print("OSCILLATION PATTERN ANALYSIS")
print("=" * 70)

# Find transitions
hit_changes = np.diff(hits.astype(int))
transitions = np.where(hit_changes != 0)[0]

print(f"\nTotal trials: {len(hits)}")
print(f"Total hits: {np.sum(hits)}")
print(f"Hit rate: {np.mean(hits):.1%}")
print(f"\nNumber of hit/miss transitions: {len(transitions)}")

# Analyze the three phases visible in the data
print("\n" + "-" * 70)
print("PHASE ANALYSIS:")
print("-" * 70)

phases = [
    ("Phase 1: Trials 0-30 (Initial misses then successes)", 0, 30),
    ("Phase 2: Trials 30-62 (Long success streak)", 30, 62),
    ("Phase 3: Trials 62-92 (Oscillating failures)", 62, 92),
]

for phase_name, start, end in phases:
    print(f"\n{phase_name}")
    phase_hits = hits[start:end]
    phase_d = d_vals[start:end]
    phase_t = t_vals[start:end]
    phase_v_hat = v_hat_vals[start:end]
    phase_sigma = sigma_v_vals[start:end]
    phase_p_pred = p_pred_vals[start:end]

    print(f"  Hit rate: {np.mean(phase_hits):.1%}")
    print(f"  Distance: mean={np.mean(phase_d):.3f}, std={np.std(phase_d):.3f}, range=[{np.min(phase_d):.2f}, {np.max(phase_d):.2f}]")
    print(f"  Time: mean={np.mean(phase_t):.3f}, std={np.std(phase_t):.3f}, range=[{np.min(phase_t):.2f}, {np.max(phase_t):.2f}]")
    print(f"  v_hat: mean={np.mean(phase_v_hat):.3f}, final={phase_v_hat[-1]:.3f}")
    print(f"  sigma_v: mean={np.mean(phase_sigma):.3f}, final={phase_sigma[-1]:.3f}")
    print(f"  Predicted P(hit): mean={np.mean(phase_p_pred):.3f}")

# ====================
# Root Cause Investigation
# ====================
print("\n" + "=" * 70)
print("ROOT CAUSE INVESTIGATION")
print("=" * 70)

print("\n1. MODE SWITCHING EFFECT (previous_hit changes success criterion)")
print("-" * 70)
print("After HIT: success = (t_pat <= t_sys) [time-based]")
print("After MISS: success = (d_pat >= d_sys) [distance-based]")
print("\nLet's trace what happens:")

# Find a transition point from success to failure
success_to_fail = np.where((hits[:-1] == 1) & (hits[1:] == 0))[0]
if len(success_to_fail) > 0:
    idx = success_to_fail[0]
    print(f"\nExample: Trial {idx} (HIT) -> Trial {idx+1} (MISS)")
    print(f"  Trial {idx}: d={d_vals[idx]:.2f}, t={t_vals[idx]:.2f}, HIT=True")
    print(f"  Trial {idx+1}: d={d_vals[idx+1]:.2f}, t={t_vals[idx+1]:.2f}, HIT=False")
    print(f"  -> Trial {idx+1} used TIME-based criterion (after hit)")
    print(f"  -> Trial {idx+2} will use DISTANCE-based criterion (after miss)")

print("\n2. SPEED ESTIMATION CONTAMINATION")
print("-" * 70)
print("The system calculates: v_obs = (dist_ratio × d_sys) / t_pat")
print("\nProblem: dist_ratio and t_pat have different meanings in hit vs miss:")

# Show examples from each mode
hit_indices = np.where(hits == 1)[0][:5]
miss_indices = np.where(hits == 0)[0][:5]

print("\nOn HITS:")
for i in hit_indices:
    dist_ratio = hist["dist_ratio"][i]
    time_ratio = hist["time_ratio"][i]
    print(f"  Trial {i}: dist_ratio={dist_ratio:.3f} (=1.0 always), time_ratio={time_ratio:.3f}")

print("\nOn MISSES (time-based mode):")
for i in miss_indices:
    if i > 0 and hits[i-1] == 1:  # Was in time-based mode
        dist_ratio = hist["dist_ratio"][i]
        time_ratio = hist["time_ratio"][i]
        print(f"  Trial {i}: dist_ratio={dist_ratio:.3f} (fell short), time_ratio={time_ratio:.3f} (=0)")

print("\n3. v_hat EVOLUTION (does it converge or oscillate?)")
print("-" * 70)

# Check for v_hat oscillation
v_hat_trend = np.diff(v_hat_vals)
increasing = np.sum(v_hat_trend > 0)
decreasing = np.sum(v_hat_trend < 0)
print(f"v_hat increases {increasing} times, decreases {decreasing} times")
print(f"Initial v_hat: {v_hat_vals[0]:.3f}")
print(f"Final v_hat: {v_hat_vals[-1]:.3f}")
print(f"Range: [{np.min(v_hat_vals):.3f}, {np.max(v_hat_vals):.3f}]")

# ====================
# Visualization
# ====================
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Plot 1: Hits over time
axes[0].scatter(range(len(hits)), hits, c=hits, cmap='RdYlGn', s=30, alpha=0.6)
axes[0].set_ylabel('Hit (1) / Miss (0)')
axes[0].set_title('Hit/Miss Pattern Over Trials')
axes[0].set_ylim(-0.1, 1.1)
axes[0].grid(True, alpha=0.3)

# Plot 2: Distance and Time
ax2 = axes[1]
ax2.plot(d_vals, 'b-', label='d_sys', alpha=0.7)
ax2.set_ylabel('Distance (m)', color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

ax2_twin = ax2.twinx()
ax2_twin.plot(t_vals, 'r-', label='t_sys', alpha=0.7)
ax2_twin.set_ylabel('Time (s)', color='r')
ax2_twin.tick_params(axis='y', labelcolor='r')
ax2_twin.legend(loc='upper right')

axes[1].set_title('Selected Distance and Time')

# Plot 3: v_hat and sigma_v
ax3 = axes[2]
ax3.plot(v_hat_vals, 'g-', label='v_hat', linewidth=2)
ax3.fill_between(range(len(v_hat_vals)),
                  v_hat_vals - 2*sigma_v_vals,
                  v_hat_vals + 2*sigma_v_vals,
                  alpha=0.3, color='g')
ax3.axhline(y=0.25, color='k', linestyle='--', label='True patient speed', alpha=0.5)
ax3.set_ylabel('Speed (m/s)')
ax3.set_title('Learned Speed Model (v_hat ± 2σ)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Predicted vs actual difficulty
axes[3].plot(p_pred_vals, 'orange', label='Predicted P(hit)', linewidth=2)
axes[3].axhline(y=0.65, color='purple', linestyle='--', label='Target p*=0.65', alpha=0.7)
axes[3].scatter(range(len(hits)), hits, c='green', s=20, alpha=0.4, label='Actual outcome')
axes[3].set_ylabel('Probability')
axes[3].set_xlabel('Trial')
axes[3].set_title('Predicted P(hit) vs Actual Outcomes')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('oscillation_analysis.png', dpi=150)
print(f"\nVisualization saved to oscillation_analysis.png")
plt.show()

# ====================
# Hypothesis Summary
# ====================
print("\n" + "=" * 70)
print("HYPOTHESES FOR OSCILLATION")
print("=" * 70)

print("""
HYPOTHESIS 1: Mode Switching Creates Different Success Criteria
  - After HIT: next trial judged on TIME (can I finish fast enough?)
  - After MISS: next trial judged on DISTANCE (can I reach far enough?)
  - These have different difficulty profiles for the same (d, t) pair
  - Creates instability in selection

HYPOTHESIS 2: Speed Estimation is Contaminated
  - Calculation: v_obs = (dist_ratio × d_sys) / t_pat
  - On hits: dist_ratio=1.0, t_pat=actual (meaningful)
  - On misses (time mode): dist_ratio<1.0, t_pat>t_sys (both reduced)
  - On misses (dist mode): dist_ratio<1.0, t_pat=sampled (mixed meaning)
  - The v_hat estimate mixes data from different regimes
  - This causes predicted P(hit) to be unreliable

HYPOTHESIS 3: Positive Feedback Loop
  1. System samples easy tasks -> many hits
  2. v_hat stays stable or increases slightly
  3. Model predicts moderate P(hit) for harder tasks
  4. Selects harder task (to match p*=0.65)
  5. Hard task fails -> switches to distance mode
  6. Distance mode has different success probability
  7. System compensates by going easier again
  8. Cycle repeats

HYPOTHESIS 4: Variability Term is Too Weak
  - w_var=0.25 vs w_eff=1.0
  - Even with normalization, effort dominates
  - System doesn't explore enough to escape local patterns
  - Gets stuck in regions that approximately match p* but are unstable

MOST LIKELY: Combination of Hypotheses 1, 2, and 3
  The mode switching + contaminated speed estimate creates an unstable
  feedback loop that the objective function cannot overcome.
""")
