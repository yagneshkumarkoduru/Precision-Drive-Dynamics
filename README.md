# Precision-Drive-Dynamics: Nonlinear Actuator Dynamics, State-Space LQR & Neural-Adaptive Sliding Mode Control

**Research Project | Advanced Nonlinear Control Systems, Non-Newtonian Friction & Physical Motion Intelligence**

[![CI](https://github.com/yagneshkumarkoduru/Precision-Drive-Dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/yagneshkumarkoduru/Precision-Drive-Dynamics/actions)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Control Theory](https://img.shields.io/badge/control-Neural--Adaptive%20STA%20%7C%20CBF%20%7C%20LESO-brightgreen.svg)](docs/paper/RESEARCH_PAPER.md)
[![Paper](https://img.shields.io/badge/paper-IEEE%20TIE%20Draft-purple.svg)](docs/paper/RESEARCH_PAPER.md)
[![Nonlinear](https://img.shields.io/badge/dynamics-Stribeck%20Friction%20%7C%20Lyapunov%20SMC-orange.svg)](docs/NONLINEAR_DYNAMICS_AND_SMC_PROOFS.md)
[![Hardware](https://img.shields.io/badge/hardware-STM32%20%7C%20FreeRTOS%20%7C%20CAN%20Bus-red.svg)](docs/IMPLEMENTATION_VERSIONS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📄 **Research Paper Manuscript:** Read the full research manuscript draft (target venue: IEEE Transactions on Industrial Electronics): [**`docs/paper/RESEARCH_PAPER.md`**](docs/paper/RESEARCH_PAPER.md) | [LaTeX Source](docs/paper/Precision_Drive_Dynamics_TIE.tex) with Theorem 1 (*Finite-Time Sliding Convergence*) and Theorem 2 (*CBF Safe Invariance*).  
> 📐 **Mathematical Derivations & Lyapunov Proofs:** Complete Moreno-Osorio quadratic Lyapunov candidate proofs, LESO error boundedness lemmas, and Stribeck models: [**`docs/NONLINEAR_DYNAMICS_AND_SMC_PROOFS.md`**](docs/NONLINEAR_DYNAMICS_AND_SMC_PROOFS.md).  
> ⚙️ **Three Implementation Tiers:** Architecture comparison and firmware for V1, V2, and V3: [**`docs/IMPLEMENTATION_VERSIONS.md`**](docs/IMPLEMENTATION_VERSIONS.md).

---

## 1. Executive Summary & Research Scope

Electromechanical actuators in high-precision robotics, surgical manipulators, and physical AI platforms must achieve sub-milliradian positioning, rapid speed settling, and immediate recovery from unknown external load torque shocks. While traditional introductory treatments assume idealized linear second-order plants governed by basic PID loops, real physical actuators exhibit **acute nonlinearities**:
- **Stribeck & Coulomb friction**: Induces stick-slip limit-cycle oscillations and steady-state deadband around zero-velocity crossovers.
- **Actuator saturation**: Voltage and current bounds cause severe integrator windup and thermal degradation.
- **Thermal parameter drift**: Armature copper resistance $R_a(T)$ shifts with temperature, while rotor inertia $J$ varies with dynamic payload grasping.

This research project develops a unified motion control framework combining:
1. **Linear Extended State Observer (LESO)**: reconstructs the lumped load disturbance from the speed measurement alone; observer bandwidth is a tuned parameter of the shared benchmark harness.
2. **Super-Twisting 2-SMC (STA) with gain adaptation**: the sliding gain grows while the tracking error is outside a dead zone and relaxes to a base value in steady state.
3. **Voltage-barrier safety filter**: closed-form solution of the state-dependent constraint $|R i_a + K_e \omega| \le V_{max}$ (48 V), intersected with the drive current limit (20 A). Because the admissible current set shrinks with speed, this is a genuine control-barrier filter rather than an output clip.
4. **Three Real-World Implementation Tiers**:
   - **Tier 1 (Embedded Microcontroller Baseline)**: MISRA-C 20 kHz PWM timer ISR for STM32 FreeRTOS paired with an ISO 11898-2 CAN-bus telemetry driver with CRC-16 integrity.
   - **Tier 2 (Stribeck LESO Hardware Driver)**: LESO state and disturbance observer with a Stribeck friction robustness view.
   - **Tier 3 (Adaptive CBF-STA)**: barrier-filtered current command with adaptive sliding gain.

**All quantitative results in this README are produced by the shared harness**
(`drive_benchmark.py`): one two-state plant, one scenario, one metric
definition set, one tuning procedure per controller class, with a 20-sample
Monte Carlo sweep. Regenerate with `python drive_benchmark.py`.

---

## 2. Mathematical Modeling & Control Synthesis

```text
       Armature Circuit                           Rotor Mechanics
      +---/\/\/\---UUUU---+
      |     R        L    |
 +    |                   |                     +----------+
v_a(t)|               ( M ) e_b(t) = K_e*w      | Rotor J  |---> w(t), theta(t)
 -    |                   |                     +----------+
      +-------------------+                          |
                                                Load T_L(t) + Friction T_f(w)
```

### 2.1 Coupled Electro-Mechanical Dynamics

$$\begin{aligned}
L_a \frac{d i_a(t)}{dt} &= V_a(t) - R_a(T) i_a(t) - K_e \omega(t) \\
J \frac{d\omega(t)}{dt} &= K_t i_a(t) - \tau_f(\omega) - \tau_L(t)
\end{aligned}$$

Where the benchmark harness uses representative 100 W-class servo parameters:
$J = 5.0\times10^{-4}\,\text{kg}\cdot\text{m}^2$, $K_t = 0.05\,\text{N}\cdot\text{m/A}$, $K_e = 0.05\,\text{V}\cdot\text{s/rad}$, $R_a = 1.0\,\Omega$, $L_a = 1.0\,\text{mH}$, $b = 5.0\times10^{-5}\,\text{N}\cdot\text{m}\cdot\text{s/rad}$, with $V_{max} = 48\,\text{V}$ and $I_{max} = 20\,\text{A}$. At the 100 rad/s operating point the back-EMF is 5 V and the 0.45 N·m load needs 9.1 A (14.1 V), so the operating point is reachable with margin; the harness asserts this before running.

### 2.2 Nonlinear Stribeck Friction Formulation

$$\tau_f(\omega) = \left[ T_c + (T_s - T_c) \exp\left(-\left(\frac{\omega}{\omega_s}\right)^2\right) \right] \operatorname{sgn}(\omega) + b_{\text{visc}} \omega$$

Where $T_c = 0.25\,\text{N}\cdot\text{m}$ (Coulomb), $T_s = 0.55\,\text{N}\cdot\text{m}$ (Static breakaway), $\omega_s = 6.0\,\text{rad/s}$ (Stribeck velocity threshold), and $b_{\text{visc}} = 0.08\,\text{N}\cdot\text{m}\cdot\text{s/rad}$.

---

## 3. Quantitative Experimental & Simulation Results

All numbers below are produced by the shared harness (`drive_benchmark.py`) on
one plant and one scenario: a 100 rad/s reference step at 0.2 s and a
0.45 N·m load step (with 0.15 N·m ripple at 20 Hz) at 1.2 s. Every controller
is tuned by the same grid procedure with the same objective (RMS tracking
error + sag + 1e-3 x steady-state chattering). Run
`python drive_benchmark.py` to regenerate; machine-readable results are in
`results/precision_drive_benchmark.json`.

| Controller (tuned) | Speed sag (rad/s) | Sag vs tuned PID | RMS tracking error (rad/s) | RMS error vs PID | Steady-state chattering (V/s) | Peak \|V\| (V) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Tuned PID | 4.12 | *baseline* | 1.63 | *baseline* | 228 | 25.0 |
| First-order SMC (boundary layer) | **0.78** | **81% lower** | 0.31 | 81% lower | 270 | 20.0 |
| Super-twisting + LESO | 1.58 | 62% lower | 0.13 | **92% lower** | 286 | 24.7 |
| Adaptive STA + voltage-barrier filter | 1.43 | 65% lower | **0.08** | **95% lower** | 282 | 24.9 |

Monte Carlo over +/-20% on J, b, Kt, R (20 samples, fixed seed):

| Controller | Sag mean +/- std (rad/s) | RMS error mean +/- std (rad/s) |
| :--- | :---: | :---: |
| Tuned PID | 4.28 +/- 0.44 | 1.63 +/- 0.09 |
| First-order SMC | 0.82 +/- 0.14 | 0.35 +/- 0.08 |
| Super-twisting + LESO | 1.58 +/- 0.30 | 0.13 +/- 0.03 |
| Adaptive STA + barrier | 1.43 +/- 0.24 | 0.08 +/- 0.01 |

### Key measured findings:
1. **Tracking quality**: the super-twisting + LESO family cuts RMS tracking
   error by 92% (fixed gain) to 95% (adaptive) against the tuned PID on the
   same plant and scenario. The adaptive variant also reduces sag by 65%
   versus PID and beats the fixed-gain STA on both metrics.
2. **Sag is not the whole story**: the first-order SMC with a small boundary
   layer achieves the smallest peak deviation (0.78 rad/s) because its
   switching term reacts within milliseconds, but it carries roughly 4x the
   residual RMS error of the STA family (boundary-layer steady-state error).
   This trade-off is reported as measured rather than claiming one controller
   dominates on every axis.
3. **Chattering**: under equal tuning with the chattering-weighted objective,
   all four controllers show comparable steady-state chattering
   (228-286 V/s RMS of dV/dt); the differences appear in the transient
   window (see `chattering_transient_v_per_s` in the results JSON).
4. **Deterministic electrical safety**: every controller, in every run
   including the Monte Carlo sweep, respects |V| <= 48 V and |i| <= 20 A.

---

## 4. Software Architecture & Directory Map

```text
Precision-Drive-Dynamics/
├── README.md                                         # Master research documentation
├── requirements.txt                                  # Python dependencies
├── classical_frequency_response.py                   # Bode, Nyquist & root locus analysis
├── modern_control_analysis.py                        # LQR, CARE & Monte Carlo robustness
├── nonlinear_friction_smc_benchmark.py               # Stribeck friction & NDOB observer
├── super_twisting_smc_eso.py                         # Super-Twisting 2-SMC & LESO benchmark
├── adaptive_cbf_super_twisting.py                    # Neural-Adaptive CBF-STA QP verification
├── docs/
│   ├── NONLINEAR_DYNAMICS_AND_SMC_PROOFS.md          # Formal Lyapunov stability & CBF proofs
│   ├── IMPLEMENTATION_VERSIONS.md                    # Architecture guide for V1, V2, and V3
│   └── paper/
│       ├── RESEARCH_PAPER.md                         # Full IEEE TIE format research draft
│       └── Precision_Drive_Dynamics_TIE.tex          # LaTeX manuscript source
├── pid_figures/                                      # Publication-grade simulation plots
│   ├── fig_adaptive_cbf_super_twisting.png           # NA-STA CBF verification
│   ├── fig_precision_drive_benchmark.png             # Tier 2 hardware benchmark
│   ├── fig_super_twisting_chattering_free.png        # Chattering suppression comparison
│   └── fig_eso_disturbance_estimation.png            # LESO reconstruction accuracy
└── implementations/                                  # Three concrete implementation versions
    ├── v1_embedded_stm32_freertos/                   # 20 kHz PWM ISR + ISO 11898 CAN HAL
    │   ├── firmware_stm32_foc.c
    │   ├── can_motor_telemetry.py
    │   └── main_embedded_drive_runner.py
    ├── v2_stribeck_leso_hardware_driver/             # Stribeck friction & 10 kHz LESO driver
    │   ├── stribeck_friction_model.py
    │   ├── high_gain_leso_driver.py
    │   └── precision_drive_benchmark.py
    └── v3_neural_adaptive_cbf_sta/                   # Neural-Adaptive CBF QP Safety Stack
        ├── neural_adaptive_super_twisting.py
        └── cbf_qp_safety_filter.py
```

---

## 5. Execution & Reproduction Guide

```bash
# Run Tier 1 Embedded CAN-Bus Telemetry Loop:
python -m implementations.v1_embedded_stm32_freertos.main_embedded_drive_runner

# Run Tier 2 Stribeck LESO Hardware Benchmark:
python -m implementations.v2_stribeck_leso_hardware_driver.precision_drive_benchmark

# Run Tier 3 Neural-Adaptive CBF-STA Verification:
python adaptive_cbf_super_twisting.py
```

---

## 6. Citation

```bibtex
@misc{koduru2026precision,
  author = {Koduru, Yagnesh Kumar},
  title  = {Neural-Adaptive Super-Twisting Sliding Mode Control with Control Barrier Functions for High-Precision Motor Drives},
  year   = {2026},
  note   = {Research manuscript draft; not peer reviewed},
  url    = {https://github.com/yagneshkumarkoduru/Precision-Drive-Dynamics}
}
```
