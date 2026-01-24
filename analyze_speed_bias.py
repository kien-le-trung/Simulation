"""
Deep dive into why v_hat from hits is so low (0.115 vs true 0.25).
"""
import numpy as np
from patient_simulation import PatientModel

print("=" * 70)
print("WHY ARE HIT SPEEDS SO LOW?")
print("=" * 70)

patient = PatientModel(seed=7)

print(f"\nPatient's natural capabilities:")
print(f"  k_d_per_sec = 0.25 m/s")
for i in range(len(patient.d_means)):
    print(f"  Level {i}: d_mean={patient.d_means[i]:.2f}m, t_mean={patient.t_means[i]:.1f}s")

print("\n" + "-" * 70)
print("SIMULATION: What speeds do we observe from hits?")
print("-" * 70)

# Test case 1: Easy tasks (should give hits)
test_cases = [
    (0.10, 5.0, "Very easy: close & slow"),
    (0.30, 4.0, "Easy: close-medium & medium-slow"),
    (0.50, 5.0, "Medium: medium & medium-slow"),
    (0.70, 7.0, "Hard: far & slow"),
]

for d_sys, t_sys, label in test_cases:
    print(f"\n{label}: d_sys={d_sys:.2f}m, t_sys={t_sys:.1f}s")
    print(f"  Required speed: {d_sys/t_sys:.4f} m/s")

    # Find distance level
    candidates = np.where(patient.d_means <= d_sys)[0]
    lvl = int(candidates[-1]) if len(candidates) > 0 else 0
    print(f"  Distance level: {lvl} (d_mean={patient.d_means[lvl]:.2f}m, t_mean={patient.t_means[lvl]:.1f}s)")

    # Run 100 trials
    hits = []
    speeds_from_hits = []
    previous_hit = True

    for _ in range(100):
        outcome = patient.sample_trial(t_sys, d_sys, lvl, previous_hit)
        hit = outcome["hit"]
        hits.append(hit)
        previous_hit = hit

        if hit:
            # What speed would we calculate?
            t_pat = outcome["t_pat"]
            v_obs = d_sys / t_pat  # This is what the code does
            speeds_from_hits.append(v_obs)

    hit_rate = np.mean(hits)
    print(f"  Hit rate: {hit_rate:.1%}")

    if len(speeds_from_hits) > 0:
        print(f"  Observed speeds from hits:")
        print(f"    Mean: {np.mean(speeds_from_hits):.4f} m/s")
        print(f"    Std:  {np.std(speeds_from_hits):.4f} m/s")
        print(f"    Min:  {np.min(speeds_from_hits):.4f} m/s")
        print(f"    Max:  {np.max(speeds_from_hits):.4f} m/s")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("""
The problem: v_obs = d_sys / t_pat

When a patient HITS:
- d_sys is the TARGET distance (what we asked for)
- t_pat is how long they ACTUALLY took

But the patient's NATURAL speed is based on (d_mean, t_mean) pairs!

Example:
- We set: d_sys=0.10m, t_sys=5.0s (very easy)
- Patient's nearest level: Level 0 (d_mean=0.25m, t_mean=1.0s)
- Patient samples: t_pat ~ N(1.0, 0.15)
- Patient finishes in: t_pat ≈ 1.0s (hits because t_pat < t_sys=5.0)
- We calculate: v_obs = 0.10 / 1.0 = 0.10 m/s

But the patient's TRUE speed is: d_mean/t_mean = 0.25/1.0 = 0.25 m/s

The issue: We're dividing a SMALL d_sys by the patient's natural t_mean!

When we set easy tasks (small d_sys), we artificially reduce the observed speed.
""")

print("\n" + "=" * 70)
print("VERIFICATION: What if we used d_mean instead of d_sys?")
print("=" * 70)

for d_sys, t_sys, label in test_cases:
    candidates = np.where(patient.d_means <= d_sys)[0]
    lvl = int(candidates[-1]) if len(candidates) > 0 else 0
    d_mean = patient.d_means[lvl]
    t_mean = patient.t_means[lvl]

    # Run trials
    speeds_corrected = []
    previous_hit = True
    for _ in range(100):
        outcome = patient.sample_trial(t_sys, d_sys, lvl, previous_hit)
        previous_hit = outcome["hit"]

        if outcome["hit"]:
            t_pat = outcome["t_pat"]
            # What if we used d_mean instead of d_sys?
            v_corrected = d_mean / t_pat
            speeds_corrected.append(v_corrected)

    if len(speeds_corrected) > 0:
        print(f"\n{label}:")
        print(f"  Using d_sys/t_pat: mean = {d_sys/t_mean:.4f} m/s (biased)")
        print(f"  Using d_mean/t_pat: mean = {np.mean(speeds_corrected):.4f} m/s (closer to 0.25)")
