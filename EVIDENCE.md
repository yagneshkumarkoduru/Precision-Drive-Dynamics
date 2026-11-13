# Evidence Register: Precision-Drive-Dynamics

Status date: 2026-11-13. Every quantitative claim in this repository traces to a
committed artifact in `results/` produced by the shared harness
(`drive_benchmark.py`) or by the NDOB-ISMC script
(`nonlinear_friction_smc_benchmark.py`). Reproduction commands are listed per
claim. Test suite: 14/14 passing (`pytest tests -q`).

## Evidence class tags

- **SIMULATION**: number produced by the numerical benchmark harness on this
  machine. No hardware was in the loop. These results are reproducible
  bit-for-bit given the committed code, fixed seeds, and the pinned
  environment, but they are not measurements of a physical drive.
- **MODEL**: analytic or parameter value stated by the plant/scenario
  definition; not an observed outcome.
- **MEASURED**: number read off physical hardware. There are currently no
  MEASURED entries; nothing in this repository has been validated on a
  physical motor.

## Plant parameters (MODEL)

Canonical plant for every shared-harness result, from
`drive_benchmark.DCMotorParams` (100 W-class brushed servo with reflected
load; documented values, not a vendor datasheet):

| Parameter | Value | Unit |
| --- | --- | --- |
| J | 5.0e-4 | kg m^2 |
| b | 5.0e-5 | N m s/rad |
| Kt | 0.05 | N m/A |
| Ke | 0.05 | V s/rad |
| R | 1.0 | ohm |
| L | 1.0e-3 | H |
| V_max | 48 | V |
| I_max | 20 | A |

Legacy analysis scripts and earlier revisions of the documentation used a
different plant (J = 0.015 kg m^2, Kt = 0.45 N m/A, 24 V bus). Those numbers
do not describe the current harness and must not be quoted for any current
claim. The standalone NDOB-ISMC script retains its own small-signal plant
(J = 0.01 kg m^2, 24 V bus); its outputs are labelled NDOB-ISMC below and are
not interchangeable with shared-harness numbers.

## Claim register (SIMULATION unless noted)

### 1. Baseline controller comparison, 100 rad/s reference, 0.45 N m load step

Artifact: `results/precision_drive_benchmark.json`.
Reproduce: `python drive_benchmark.py`.

| Controller | Speed sag (rad/s) | RMS tracking error (rad/s) | Steady-state chattering (V/s) |
| --- | --- | --- | --- |
| Tuned PID | 4.12 | 1.63 | 228 |
| First-order SMC | 0.78 | 0.31 | 270 |
| Super-twisting + LESO | 1.58 | 0.13 | 286 |
| Adaptive STA + voltage-barrier filter | 1.43 | 0.08 | 282 |

Derived SIMULATION claims: super-twisting reduces RMS tracking error by 92%
vs tuned PID; the adaptive variant by 95%. Adaptive variant reduces sag by
65% vs tuned PID (1.43 vs 4.12 rad/s).

### 2. Monte Carlo robustness, +/-20% on J, b, Kt, R (20 samples, seed 20261011)

Artifact: same JSON, `monte_carlo` block. SIMULATION.

| Controller | Sag mean +/- std (rad/s) | RMS error mean +/- std (rad/s) |
| --- | --- | --- |
| Tuned PID | 4.28 +/- 0.44 | 1.63 +/- 0.09 |
| First-order SMC | 0.82 +/- 0.14 | 0.35 +/- 0.08 |
| Super-twisting + LESO | 1.58 +/- 0.30 | 0.13 +/- 0.03 |
| Adaptive STA + CBF | 1.43 +/- 0.24 | 0.08 +/- 0.01 |

### 3. Multi-operating-point sweep with chattering FFT

Artifact: `results/strengthened_benchmark.json`.
Reproduce: `python strengthened_benchmark.py`. SIMULATION.

Seven operating points (w_ref in {50, 100, 150} rad/s crossed with load steps
in {0.20, 0.45, 0.70} N m), gains tuned once at 100 rad/s / 0.45 N m and
reused everywhere. Per controller per point: sag, RMS error, recovery,
steady-state chattering RMS, dominant chattering frequency and spectral
centroid from an FFT of dV/dt over the final 1.0 s window (Hann window), peak
voltage, peak current.

Headline SIMULATION findings:

- At the tuning point the controllers reproduce the baseline table above
  exactly, confirming consistency between artifacts.
- At 0.70 N m (heaviest load), adaptive STA + CBF holds sag to 2.72 rad/s vs
  6.32 rad/s for tuned PID (57% lower, SIMULATION). Fixed-gain first-order
  SMC saturates at this load (sag 6.99 rad/s, worse than PID) while chattering
  climbs to 1059 V/s with a dominant spectral peak at 180 Hz; the adaptive
  variant stays at 485 V/s near the 20 Hz load-ripple line.
- Dominant chattering frequency is 20 Hz across most controllers and points,
  i.e. the metric is dominated by the load-ripple response rather than
  switching noise; the FFT columns separate the two effects.

### 4. Stribeck-friction scenario (shared harness)

Artifact: `results/precision_drive_stribeck.json`.
Reproduce: `python implementations/v2_stribeck_leso_hardware_driver/precision_drive_benchmark.py`.
SIMULATION. Adding Stribeck friction (T_c = 0.02 N m, T_s = 0.05 N m,
omega_s = 5 rad/s) changes sag by less than 0.02 rad/s for every controller
at the 100 rad/s operating point; the ranking is unchanged.

### 5. NDOB-ISMC low-speed Stribeck study (standalone plant)

Artifact: console output of `nonlinear_friction_smc_benchmark.py`, figures
`pid_figures/fig12_stribeck_friction_tracking.png` and
`fig13_ndob_disturbance_reconstruction.png`. SIMULATION on the standalone
small-signal plant (J = 0.01 kg m^2, 24 V bus, 0.5 rad/s reference, 0.025 N m
shock load between 1.5 s and 2.2 s).

- Disturbance sag attenuation: 21.6% (PID sag 0.0672 rad/s, NDOB-ISMC sag
  0.0527 rad/s). SIMULATION.
- RMS tracking error reduction: 5.2% (0.1076 to 0.1020 rad/s). SIMULATION.

### 6. Analytic results (MODEL)

`docs/NONLINEAR_DYNAMICS_AND_SMC_PROOFS.md` contains Lyapunov stability
arguments and observer convergence derivations. These are MODEL-class
statements about the idealized plant and controller equations; they carry no
numerical performance claim and are not hardware evidence.

## Claims explicitly withdrawn

The following legacy figures appeared in earlier revisions and are invalid
under the current harness. Do not quote them:

- Tier 2 disturbance sag reduction of 56.59% (derived from the retired
  J = 0.015 kg m^2 plant; superseded by claim 5).
- Tier 3 sag reduction of 90.9% (14.8 to 1.35 rad/s; retired plant and
  scenario; superseded by claims 1 and 3).
- "100% boundary cut" and "95.4% chattering variance drop" (no committed
  artifact reproduces them).

## Test status

14/14 tests passing as of 2026-11-13 (`pytest tests -q`). Tests cover the
shared harness metric definitions, controller saturation behaviour, and the
tuning cache; they are consistency checks, not hardware validation.
