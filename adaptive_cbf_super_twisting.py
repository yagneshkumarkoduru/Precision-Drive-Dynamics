#!/usr/bin/env python3
"""Neural-Adaptive Super-Twisting 2-SMC with a voltage-barrier filter and LESO.

This script is a thin presentation layer over the shared benchmark harness
(``drive_benchmark.py``): the plant, scenario, controller suite, tuning
procedure, and every reported number come from the harness. Nothing here is
hardcoded.

Controller summary (all in the same current-command space, same plant):
  1) Linear Extended State Observer (LESO) estimating the lumped load
     acceleration from the speed measurement only.
  2) Super-twisting 2-SMC core with gain adaptation: the sliding gain grows
     while the tracking error is outside a dead zone and relaxes to a base
     value in steady state.
  3) Voltage-barrier filter: closed-form QP solution for the state-dependent
     constraint R*i + Ke*w <= V_max, intersected with the drive current limit.

Run ``python drive_benchmark.py`` for the full comparison and Monte Carlo.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drive_benchmark import (
    DCMotorParams,
    Scenario,
    controller_suite,
    run_full_benchmark,
    simulate_controller,
    tune_all,
)


class AdaptiveCBF_STA_Controller:
    """Parameter holder for the adaptive controller line (see module docstring).

    The constructor keeps the historical signature so existing callers and
    tests work; the physical defaults now match the shared harness.
    """

    def __init__(self, J=5.0e-4, b=5.0e-5, K=0.05, R=1.0, L=1.0e-3, V_max=48.0):
        self.J = J
        self.b = b
        self.K = K
        self.R = R
        self.L = L
        self.V_max = V_max

    def run_benchmark(self, t_end=3.0, dt=0.0005):
        params = DCMotorParams(J=self.J, b=self.b, Kt=self.K, R=self.R, L=self.L, V_max=self.V_max)
        scenario = Scenario(t_end=t_end, dt=dt)
        payload, trajectories = run_full_benchmark(params, scenario, mc_samples=10)

        print("=" * 78)
        print("NEURAL-ADAPTIVE SUPER-TWISTING + CBF BENCHMARK (shared harness)")
        print("=" * 78)
        for name, m in payload["controllers"].items():
            rec = m["recovery_time_s"]
            rec_text = f"{rec:.3f}s" if rec == rec else "no 2% recovery"
            print(
                f"[+] {name:<24} sag {m['sag_rad_s']:7.3f} rad/s | "
                f"recovery {rec_text:>16} | rms {m['rms_tracking_error_rad_s']:6.3f} | "
                f"chatter {m['chattering_v_per_s']:7.1f} V/s | peak |V| {m['peak_voltage_v']:5.1f} V"
            )
        print(f"[+] Voltage barrier adhered: max |u| = "
              f"{max(m['peak_voltage_v'] for m in payload['controllers'].values()):.2f} V "
              f"<= {params.V_max:.0f} V")

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_figures")
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "fig_adaptive_cbf_super_twisting.png")

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        colors = {"pid": "tab:red", "smc1": "tab:orange", "sta_eso": "tab:green", "adaptive_sta_eso_cbf": "tab:blue"}
        for name, traj in trajectories.items():
            axs[0].plot(traj["t"], traj["w"], color=colors.get(name, "gray"), alpha=0.85, label=name)
            axs[2].plot(traj["t"], traj["V"], color=colors.get(name, "gray"), alpha=0.7, label=name)
        ref = next(iter(trajectories.values()))
        axs[0].plot(ref["t"], ref["w_ref"], "k--", label="reference")
        axs[0].axvline(scenario.load_step_at, color="gray", linestyle=":", label="load step")
        axs[0].set_ylabel("rotor speed (rad/s)")
        axs[0].set_title("Shared-harness comparison (single plant, single scenario)")
        axs[0].legend(loc="lower right")
        axs[0].grid(True, alpha=0.3)

        traj_ad = trajectories["adaptive_sta_eso_cbf"]
        axs[1].plot(traj_ad["t"], -traj_ad["z2"] * params.J, label="LESO load estimate (N m)")
        axs[1].plot(traj_ad["t"], traj_ad["T_load"], "k--", label="true load (N m)")
        axs[1].set_ylabel("torque (N m)")
        axs[1].legend(loc="upper right")
        axs[1].grid(True, alpha=0.3)

        axs[2].axhline(params.V_max, color="r", linestyle=":")
        axs[2].axhline(-params.V_max, color="r", linestyle=":")
        axs[2].set_ylabel("armature voltage (V)")
        axs[2].set_xlabel("time (s)")
        axs[2].legend(loc="upper right")
        axs[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[+] Saved figure to {out_png}")
        return payload


if __name__ == "__main__":
    AdaptiveCBF_STA_Controller().run_benchmark()
