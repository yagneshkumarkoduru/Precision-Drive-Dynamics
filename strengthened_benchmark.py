#!/usr/bin/env python3
"""Strengthened benchmark for the precision DC drive study.

Extends the shared harness (drive_benchmark.py) with three protocol upgrades:

1. Multi-operating-point disturbance rejection: the controller suite is
   evaluated at three reference speeds (50, 100, 150 rad/s) and three load
   step magnitudes (0.20, 0.45, 0.70 N m), not a single step. Every
   combination reuses the same tuned gains (no per-point re-tuning).
2. Chattering frequency analysis: an FFT of the steady-state voltage command
   reports the dominant chattering frequency and the spectral centroid, in
   addition to the RMS dV/dt metric already in the harness.
3. A single committed artifact (results/strengthened_benchmark.json) that
   records the plant, the tuned gains, the disturbance profile, and every
   metric per controller per operating point.

Run:  python strengthened_benchmark.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from drive_benchmark import (
    DCMotorParams,
    RESULTS_DIR,
    Scenario,
    compute_metrics,
    controller_suite,
    simulate_controller,
    tune_all,
)

ROOT = Path(__file__).resolve().parent

OPERATING_POINTS = [
    {"w_ref": 50.0, "load_nm": 0.20},
    {"w_ref": 50.0, "load_nm": 0.45},
    {"w_ref": 100.0, "load_nm": 0.20},
    {"w_ref": 100.0, "load_nm": 0.45},
    {"w_ref": 100.0, "load_nm": 0.70},
    {"w_ref": 150.0, "load_nm": 0.45},
    {"w_ref": 150.0, "load_nm": 0.70},
]


def chattering_spectrum(V: np.ndarray, dt: float, steady_mask: np.ndarray) -> dict[str, float]:
    """FFT of the steady-state voltage command.

    Returns the dominant spectral peak frequency and the spectral centroid of
    |FFT(dV/dt)| over the steady window. Both are SIMULATION quantities derived
    from the harness trajectories.
    """
    v = V[steady_mask]
    if v.size < 16:
        return {"dominant_hz": float("nan"), "spectral_centroid_hz": float("nan")}
    dv = np.diff(v) / dt
    dv = dv - np.mean(dv)
    window = np.hanning(dv.size)
    spec = np.abs(np.fft.rfft(dv * window))
    freqs = np.fft.rfftfreq(dv.size, d=dt)
    if spec.sum() <= 0.0:
        return {"dominant_hz": 0.0, "spectral_centroid_hz": 0.0}
    dominant = float(freqs[int(np.argmax(spec))])
    centroid = float(np.sum(freqs * spec) / np.sum(spec))
    return {"dominant_hz": dominant, "spectral_centroid_hz": centroid}


def run_point(params: DCMotorParams, gains: dict, w_ref: float, load_nm: float) -> dict[str, dict]:
    sc = Scenario(w_ref_value=w_ref, load_step_nm=load_nm, load_ripple_nm=0.15 * load_nm / 0.45)
    suite = controller_suite(params, gains)
    out: dict[str, dict] = {}
    for name, (ctrl, use_barrier) in suite.items():
        traj = simulate_controller(ctrl, params, sc, use_barrier=use_barrier)
        m = compute_metrics(traj, sc)
        ss_mask = traj["t"] >= max(sc.load_step_at + 0.3, sc.t_end - 1.0)
        spec = chattering_spectrum(traj["V"], sc.dt, ss_mask)
        m.update(spec)
        m["peak_voltage_v"] = float(np.max(np.abs(traj["V"])))
        m["peak_current_a"] = float(np.max(np.abs(traj["i"])))
        out[name] = m
    return out


def main() -> int:
    params = DCMotorParams()
    gains = tune_all(params)

    points: list[dict] = []
    for op in OPERATING_POINTS:
        res = run_point(params, gains, op["w_ref"], op["load_nm"])
        points.append({"operating_point": op, "controllers": res})
        pid_sag = res["pid"]["sag_rad_s"]
        print(f"w_ref={op['w_ref']:6.1f}  load={op['load_nm']:.2f} N m")
        for name, m in res.items():
            ratio = m["sag_rad_s"] / pid_sag if pid_sag > 0 else float("nan")
            print(
                f"  {name:<24} sag {m['sag_rad_s']:6.3f} ({ratio:5.2f}x pid)  "
                f"rms {m['rms_tracking_error_rad_s']:6.3f}  "
                f"chatter {m['chattering_v_per_s']:7.1f} V/s @ {m['dominant_hz']:7.1f} Hz"
            )

    payload = {
        "plant": asdict(params),
        "gains": gains,
        "protocol": {
            "operating_points": OPERATING_POINTS,
            "chattering_spectrum": "FFT of dV/dt over the final 1.0 s steady window, Hann window",
            "note": "gains tuned once on the 100 rad/s, 0.45 N m reference point; reused everywhere",
        },
        "points": points,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "strengthened_benchmark.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
