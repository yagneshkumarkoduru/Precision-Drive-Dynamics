"""
Super-Twisting Second-Order Sliding Mode Control (STA-2SMC) & LESO presentation layer.

All physics, tuning, and numbers come from the shared harness
(``drive_benchmark.py``). The legacy ``simulate`` interface is preserved for
callers and tests; it returns the harness trajectories for the first-order SMC
and the super-twisting + ESO pair.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drive_benchmark import (
    DCMotorParams,
    Scenario,
    controller_suite,
    run_full_benchmark,
    simulate_controller,
    tune_all,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35


class SuperTwistingSMC_ESO_Engine:
    """Thin wrapper over the shared harness controllers."""

    def __init__(self) -> None:
        self.params = DCMotorParams()

    def simulate(self, t_end: float = 3.0, dt: float = 5.0e-4):
        """Run the SMC1 and STA+ESO pair on the shared scenario.

        Returns (t, w_ref, T_dist, w_smc1, u_smc1, w_sta, u_sta, z2_eso) with
        the same shapes as the historical implementation.
        """
        scenario = Scenario(t_end=t_end, dt=dt)
        gains = tune_all(self.params)
        suite = controller_suite(self.params, gains)
        traj_smc = simulate_controller(suite["smc1"][0], self.params, scenario)
        traj_sta = simulate_controller(suite["sta_eso"][0], self.params, scenario)
        z2 = traj_sta.get("z2", np.zeros_like(traj_sta["t"]))
        return (
            traj_sta["t"],
            traj_sta["w_ref"],
            traj_sta["T_load"],
            traj_smc["w"],
            traj_smc["V"],
            traj_sta["w"],
            traj_sta["V"],
            z2,
        )

    def generate_figures(self):
        payload, trajectories = run_full_benchmark(self.params, Scenario(), mc_samples=10)
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_figures")
        os.makedirs(out_dir, exist_ok=True)

        traj_smc = trajectories["smc1"]
        traj_sta = trajectories["sta_eso"]

        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)
        ax1.plot(traj_sta["t"], traj_sta["w_ref"], "k--", label="reference")
        ax1.plot(traj_smc["t"], traj_smc["w"], "r-", alpha=0.8, label="first-order SMC")
        ax1.plot(traj_sta["t"], traj_sta["w"], "b-", label="super-twisting + ESO")
        ax1.axvline(Scenario().load_step_at, color="gray", linestyle=":", label="load step")
        ax1.set_ylabel("rotor speed (rad/s)")
        ax1.set_title("Shared-harness speed tracking (measured)")
        ax1.legend(loc="lower right")
        ax2.plot(traj_smc["t"], traj_smc["V"], "r-", alpha=0.7, label="first-order SMC voltage")
        ax2.plot(traj_sta["t"], traj_sta["V"], "b-", label="super-twisting voltage")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("armature voltage (V)")
        fig1.tight_layout()
        p1 = os.path.join(out_dir, "fig_super_twisting_chattering_free.png")
        fig1.savefig(p1, dpi=200)
        plt.close(fig1)

        fig2, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(traj_sta["t"], traj_sta["T_load"], "r-", label="true load torque")
        ax.plot(traj_sta["t"], -traj_sta["z2"] * self.params.J, "b--", label="LESO estimate")
        ax.axvline(Scenario().load_step_at, color="gray", linestyle=":", label="load step")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("load torque (N m)")
        ax.set_title("Sensorless LESO load-torque reconstruction (measured)")
        ax.legend(loc="upper left")
        fig2.tight_layout()
        p2 = os.path.join(out_dir, "fig_eso_disturbance_estimation.png")
        fig2.savefig(p2, dpi=200)
        plt.close(fig2)
        return p1, p2


def run_super_twisting_benchmark():
    print("=" * 78)
    print("SUPER-TWISTING 2-SMC & LESO BENCHMARK (shared harness; measured numbers)")
    print("=" * 78)
    engine = SuperTwistingSMC_ESO_Engine()
    payload, _ = run_full_benchmark(engine.params, Scenario(), mc_samples=10)
    for name, m in payload["controllers"].items():
        print(
            f"  {name:<24} sag {m['sag_rad_s']:7.3f} rad/s  "
            f"rms {m['rms_tracking_error_rad_s']:6.3f} rad/s  "
            f"chatter {m['chattering_v_per_s']:7.1f} V/s"
        )
    p1, p2 = engine.generate_figures()
    print(f"[OK] Figures: {p1} | {p2}")
    print("=" * 78)


if __name__ == "__main__":
    run_super_twisting_benchmark()
