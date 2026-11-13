"""Shared-harness tests: plant, metrics, controller relations, barrier, tuning cache.

These tests pin the honest measured relations produced by drive_benchmark.py
(cached gains from results/tuned_gains.json). They assert only what the
measured data shows, with margins, so they fail if the harness regresses.
"""

import numpy as np
import pytest

pytest.importorskip("numpy")

from drive_benchmark import (
    DCMotorParams,
    PIDController,
    Scenario,
    SuperTwistingESO,
    VoltageBarrierFilter,
    compute_metrics,
    controller_suite,
    run_full_benchmark,
    simulate_controller,
    tune_all,
)

PARAMS = DCMotorParams()
SCENARIO = Scenario()


def _gains():
    return tune_all(PARAMS)


def _run(name: str):
    gains = _gains()
    suite = controller_suite(PARAMS, gains)
    ctrl, barrier = suite[name]
    return simulate_controller(ctrl, PARAMS, SCENARIO, use_barrier=barrier)


def test_plant_reaches_reference_and_respects_limits():
    traj = _run("pid")
    t, w = traj["t"], traj["w"]
    steady = (t >= 2.0) & (t <= 2.95)
    assert np.mean(np.abs(w[steady] - SCENARIO.w_ref_value)) < 1.0
    assert np.max(np.abs(traj["V"])) <= PARAMS.V_max + 1e-9
    assert np.max(np.abs(traj["i"])) <= PARAMS.I_max + 1e-6


def test_all_controllers_respect_voltage_and_current_limits():
    gains = _gains()
    for name, (ctrl, barrier) in controller_suite(PARAMS, gains).items():
        traj = simulate_controller(ctrl, PARAMS, SCENARIO, use_barrier=barrier)
        assert np.max(np.abs(traj["V"])) <= PARAMS.V_max + 1e-9, name
        assert np.max(np.abs(traj["i"])) <= PARAMS.I_max + 1e-6, name


def test_metrics_are_finite_and_ordered():
    m = compute_metrics(_run("sta_eso"), SCENARIO)
    for key in ("sag_rad_s", "rms_tracking_error_rad_s", "chattering_v_per_s",
                "chattering_transient_v_per_s", "peak_voltage_v", "peak_current_a"):
        assert np.isfinite(m[key]), key
        assert m[key] >= 0.0, key
    assert m["sag_rad_s"] < 10.0
    assert m["rms_tracking_error_rad_s"] < 5.0


def test_super_twisting_beats_pid_on_rms_and_sag():
    m_pid = compute_metrics(_run("pid"), SCENARIO)
    m_sta = compute_metrics(_run("sta_eso"), SCENARIO)
    assert m_sta["rms_tracking_error_rad_s"] < 0.5 * m_pid["rms_tracking_error_rad_s"]
    assert m_sta["sag_rad_s"] < 0.7 * m_pid["sag_rad_s"]


def test_adaptive_matches_or_beats_fixed_sta_on_rms():
    m_sta = compute_metrics(_run("sta_eso"), SCENARIO)
    m_ad = compute_metrics(_run("adaptive_sta_eso_cbf"), SCENARIO)
    assert m_ad["rms_tracking_error_rad_s"] <= m_sta["rms_tracking_error_rad_s"] * 1.05
    assert m_ad["sag_rad_s"] <= m_sta["sag_rad_s"] * 1.1


def test_monte_carlo_is_deterministic():
    payload_a, _ = run_full_benchmark(PARAMS, SCENARIO, mc_samples=2)
    payload_b, _ = run_full_benchmark(PARAMS, SCENARIO, mc_samples=2)
    for name in payload_a["controllers"]:
        assert payload_a["monte_carlo"]["summary"][name]["sag_mean"] == pytest.approx(
            payload_b["monte_carlo"]["summary"][name]["sag_mean"], rel=0, abs=0
        )


def test_barrier_shrinks_with_speed_and_projects():
    barrier = VoltageBarrierFilter(PARAMS)
    i_low = barrier.admissible_current(0.0)
    i_high = barrier.admissible_current(100.0)
    assert i_low > i_high > 0.0
    assert barrier.filter(1e6, 100.0) == pytest.approx(min(i_high, PARAMS.I_max))
    assert barrier.filter(-1e6, 100.0) == pytest.approx(-min(i_high, PARAMS.I_max))


def test_tuning_cache_is_stable():
    a = tune_all(PARAMS)
    b = tune_all(PARAMS)
    assert a == b
