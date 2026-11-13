"""
Precision Drive Dynamics: Stribeck-friction robustness view of the shared harness.

The plant, scenario, controller suite, and tuning all come from
``drive_benchmark.py``. This script re-evaluates the same tuned controllers on
a scenario with Stribeck friction enabled and writes one figure plus a JSON
summary. No numbers are hardcoded.

    python implementations/v2_stribeck_leso_hardware_driver/precision_drive_benchmark.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from drive_benchmark import (  # noqa: E402
    RESULTS_DIR,
    Scenario,
    compute_metrics,
    controller_suite,
    simulate_controller,
    tune_all,
)


def run_precision_drive_benchmark():
    params = None  # default harness plant
    from drive_benchmark import DCMotorParams
    params = DCMotorParams()

    # Same scenario plus Stribeck friction at the contact.
    scenario = Scenario(stribeck_tc_nm=0.02, stribeck_ts_nm=0.05, stribeck_ws_rad_s=5.0)

    # Gains come from the shared tuning cache (see drive_benchmark.tune_all);
    # this study measures friction robustness, it does not re-tune.
    gains = tune_all(params)
    suite = controller_suite(params, gains)

    results = {}
    trajectories = {}
    for name, (ctrl, use_barrier) in suite.items():
        traj = simulate_controller(ctrl, params, scenario, use_barrier=use_barrier)
        trajectories[name] = traj
        results[name] = compute_metrics(traj, scenario)

    print("=" * 78)
    print("PRECISION DRIVE: STRIBECK-FRICTION SCENARIO (shared harness)")
    print("=" * 78)
    header = f"{'controller':<24} {'sag(rad/s)':>10} {'rms_err':>9} {'chatter(V/s)':>13}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<24} {m['sag_rad_s']:>10.3f} {m['rms_tracking_error_rad_s']:>9.3f} {m['chattering_v_per_s']:>13.1f}")
    print("-" * len(header))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "precision_drive_stribeck.json"
    out_json.write_text(json.dumps({"scenario": scenario.__dict__, "results": results}, indent=2) + "\n", encoding="utf-8")

    fig_dir = _ROOT / "pid_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = {"pid": "tab:red", "smc1": "tab:orange", "sta_eso": "tab:green", "adaptive_sta_eso_cbf": "tab:blue"}
    ref = trajectories["pid"]
    axs[0].plot(ref["t"], ref["w_ref"], "k--", label="reference")
    for name, traj in trajectories.items():
        axs[0].plot(traj["t"], traj["w"], color=colors.get(name, "gray"), alpha=0.85, label=name)
    axs[0].axvline(scenario.load_step_at, color="gray", linestyle=":", label="load step")
    axs[0].set_ylabel("speed (rad/s)")
    axs[0].set_title("Stribeck-friction scenario (measured, shared harness)")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(ref["t"], ref["T_load"], "k-", label="load + Stribeck torque (N m)")
    axs[1].set_ylabel("torque (N m)")
    axs[1].legend(loc="upper left")
    axs[1].grid(True, alpha=0.3)

    for name, traj in trajectories.items():
        axs[2].plot(traj["t"], traj["V"], color=colors.get(name, "gray"), alpha=0.7, label=name)
    axs[2].axhline(48.0, color="r", linestyle=":")
    axs[2].axhline(-48.0, color="r", linestyle=":")
    axs[2].set_ylabel("voltage (V)")
    axs[2].set_xlabel("time (s)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = fig_dir / "fig_precision_drive_benchmark.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"figure: {out_png}")
    print(f"written: {out_json}\n")
    return results


if __name__ == "__main__":
    run_precision_drive_benchmark()
