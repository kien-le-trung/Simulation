"""
Diagnostic script to investigate why d=0.8m, t=3-3.5s is being selected.
Tests both the patient model and the objective function.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from operations_research import (
    score_candidate, p_hit_from_speed, bin9,
    D_MIN, D_MAX, T_MIN, T_MAX, CANDIDATES
)
from patient_simulation import PatientModel

# ====================
# Test 1: Patient Model Analysis
# ====================
def test_patient_model():
    """Check if patient model makes d=0.8, t=3-3.5 naturally successful."""
    print("=" * 60)
    print("TEST 1: Patient Model Capability Analysis")
    print("=" * 60)

    patient = PatientModel(seed=7)
    print(f"\nPatient's natural (d_mean, t_mean) pairs:")
    print(f"  k_d_per_sec = {patient.d_means[1]/patient.t_means[1]:.3f} m/s")
    for i, (d, t) in enumerate(zip(patient.d_means, patient.t_means)):
        v = d / t
        print(f"  Level {i}: d={d:.2f}m, t={t:.1f}s, v={v:.3f} m/s")

    # Test specific combinations
    print("\n" + "-" * 60)
    print("Simulating 1000 trials for key (d_sys, t_sys) combinations:")
    print("-" * 60)

    test_cases = [
        (0.80, 1.0, "Far & Fast"),
        (0.80, 3.0, "Far & Medium-Fast"),
        (0.80, 3.5, "Far & Medium"),
        (0.80, 7.0, "Far & Slow"),
        (0.40, 3.0, "Medium & Medium-Fast"),
        (0.10, 3.0, "Close & Medium-Fast"),
    ]

    results = []
    for d_sys, t_sys, label in test_cases:
        # Find distance level
        candidates = np.where(patient.d_means <= d_sys)[0]
        lvl = int(candidates[-1]) if len(candidates) > 0 else 0

        hits = []
        previous_hit = True
        for _ in range(1000):
            outcome = patient.sample_trial(t_sys, d_sys, lvl, previous_hit)
            hits.append(outcome["hit"])
            previous_hit = outcome["hit"]

        hit_rate = np.mean(hits)
        v_req = d_sys / t_sys
        results.append((label, d_sys, t_sys, v_req, hit_rate))
        print(f"{label:20s}: d={d_sys:.2f}m, t={t_sys:.1f}s, "
              f"v_req={v_req:.3f} m/s -> Hit rate: {hit_rate:.1%}")

    return results


# ====================
# Test 2: Objective Function Analysis
# ====================
def test_objective_function():
    """Check what the objective function prefers at different trial stages."""
    print("\n" + "=" * 60)
    print("TEST 2: Objective Function Preference Analysis")
    print("=" * 60)

    # Simulate different stages of learning
    stages = [
        ("Early (trial 10)", 10, 0.25, 0.15),
        ("Mid (trial 100)", 100, 0.27, 0.08),
        ("Late (trial 250)", 250, 0.28, 0.06),
    ]

    for stage_name, n_trials, v_hat, sigma_v in stages:
        print(f"\n{stage_name}: v_hat={v_hat:.3f}, sigma_v={sigma_v:.3f}")
        print("-" * 60)

        # Create mock counts (simulate uniform-ish sampling)
        counts_3x3 = np.ones((3, 3), dtype=int) * (n_trials // 9)

        # Score all candidates
        scores = []
        for d, t in CANDIDATES:
            sc, p = score_candidate(
                d, t,
                v_hat=v_hat, sigma_v=sigma_v,
                p_star=0.65,
                counts_3x3=counts_3x3,
                d_prev=0.40, t_prev=4.0,  # some arbitrary previous
                w_eff=1.0, w_var=0.25, w_smooth=0.40,
                p_min=0.10
            )
            if sc > -1e8:  # valid candidate
                scores.append((d, t, sc, p))

        # Sort by score
        scores.sort(key=lambda x: x[2], reverse=True)

        # Show top 10
        print(f"{'Rank':<6}{'d (m)':<8}{'t (s)':<8}{'v_req':<10}{'P(hit)':<10}{'Score':<10}{'Bin'}")
        for i, (d, t, sc, p) in enumerate(scores[:10]):
            v_req = d / t
            bin_label = bin9(d, t)
            print(f"{i+1:<6}{d:<8.2f}{t:<8.2f}{v_req:<10.3f}{p:<10.3f}{sc:<10.4f}{bin_label}")

    return scores


# ====================
# Test 3: Speed Estimation Bias
# ====================
def test_speed_estimation():
    """Check if speed estimation creates bias."""
    print("\n" + "=" * 60)
    print("TEST 3: Speed Estimation from Patient Data")
    print("=" * 60)

    patient = PatientModel(seed=7)

    # Simulate collecting speed data from various (d,t) combinations
    print("\nSimulating what v_patient would learn from different sampling strategies:")

    strategies = [
        ("Far & Fast focused", [(0.80, 3.0)] * 50),
        ("Uniform random", [(d, t) for d in [0.2, 0.4, 0.6, 0.8] for t in [2.0, 4.0, 6.0]]),
        ("Close & Slow focused", [(0.20, 5.0)] * 50),
    ]

    for strategy_name, tasks in strategies:
        v_patient = []
        previous_hit = True

        for d_sys, t_sys in tasks:
            candidates = np.where(patient.d_means <= d_sys)[0]
            lvl = int(candidates[-1]) if len(candidates) > 0 else 0

            outcome = patient.sample_trial(t_sys, d_sys, lvl, previous_hit)
            d_pat = float(outcome["dist_ratio"]) * d_sys
            t_pat = float(outcome["t_pat"])
            v_obs = d_pat / max(t_pat, 1e-6)
            v_patient.append(v_obs)
            previous_hit = outcome["hit"]

        v_hat = np.mean(v_patient)
        sigma_v = np.std(v_patient, ddof=1) if len(v_patient) > 1 else 0.0

        print(f"\n{strategy_name}:")
        print(f"  Learned: v_hat={v_hat:.3f} m/s, sigma_v={sigma_v:.3f} m/s")

        # What does this predict for key combinations?
        test_combos = [(0.80, 3.0), (0.40, 4.0), (0.20, 5.0)]
        print(f"  Predicted P(hit) for:")
        for d, t in test_combos:
            p = p_hit_from_speed(d, t, v_hat, sigma_v)
            v_req = d / t
            print(f"    (d={d:.2f}, t={t:.1f}, v_req={v_req:.3f}): P={p:.3f}")


# ====================
# Test 4: Visualize Score Landscape
# ====================
def visualize_score_landscape():
    """Create heatmap of scores across (d,t) space."""
    print("\n" + "=" * 60)
    print("TEST 4: Score Landscape Visualization")
    print("=" * 60)

    # Mid-trial conditions
    v_hat, sigma_v = 0.27, 0.08
    counts_3x3 = np.ones((3, 3), dtype=int) * 30

    # Create grid
    d_vals = np.arange(D_MIN, D_MAX + 0.01, 0.05)
    t_vals = np.arange(T_MIN, T_MAX + 0.01, 0.25)

    score_grid = np.zeros((len(t_vals), len(d_vals)))
    p_grid = np.zeros((len(t_vals), len(d_vals)))

    for i, t in enumerate(t_vals):
        for j, d in enumerate(d_vals):
            sc, p = score_candidate(
                d, t,
                v_hat=v_hat, sigma_v=sigma_v,
                p_star=0.65,
                counts_3x3=counts_3x3,
                d_prev=None, t_prev=None,
                w_eff=1.0, w_var=0.25, w_smooth=0.40,
                p_min=0.10
            )
            score_grid[i, j] = sc if sc > -1e8 else np.nan
            p_grid[i, j] = p

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Score heatmap
    im1 = ax1.imshow(score_grid, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Objective Score Landscape')
    ax1.set_xticks(np.arange(0, len(d_vals), 4))
    ax1.set_xticklabels([f'{d:.2f}' for d in d_vals[::4]])
    ax1.set_yticks(np.arange(0, len(t_vals), 4))
    ax1.set_yticklabels([f'{t:.1f}' for t in t_vals[::4]])
    plt.colorbar(im1, ax=ax1, label='Score')

    # P(hit) heatmap
    im2 = ax2.imshow(p_grid, aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Predicted P(hit)')
    ax2.set_xticks(np.arange(0, len(d_vals), 4))
    ax2.set_xticklabels([f'{d:.2f}' for d in d_vals[::4]])
    ax2.set_yticks(np.arange(0, len(t_vals), 4))
    ax2.set_yticklabels([f'{t:.1f}' for t in t_vals[::4]])
    plt.colorbar(im2, ax=ax2, label='P(hit)')

    # Mark the observed preference region
    ax1.axhline(y=8, color='red', linestyle='--', linewidth=2, label='t=3.0-3.5s')
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=14, color='red', linestyle='--', linewidth=2, label='d=0.8m')
    ax1.legend()

    plt.tight_layout()
    plt.savefig('score_landscape.png', dpi=150)
    print("Saved visualization to score_landscape.png")
    plt.show()


if __name__ == "__main__":
    # Run all tests
    patient_results = test_patient_model()
    obj_results = test_objective_function()
    test_speed_estimation()
    visualize_score_landscape()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nCheck the results above and score_landscape.png to identify")
    print("whether the bias is from:")
    print("  1. Patient model (high success rate at d=0.8, t=3-3.5)")
    print("  2. Objective function (high scores for that region)")
    print("  3. Speed estimation bias (learning wrong v_hat)")
