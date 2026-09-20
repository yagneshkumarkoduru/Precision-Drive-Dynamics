"""Deterministic drive-control tests (no hardware, run in CI).

Covers the Stribeck friction plant, the nonlinear disturbance observer and
integral sliding-mode controller, the super-twisting with ESO engine, and the
state-space design checks. All numeric and deterministic.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from adaptive_cbf_super_twisting import AdaptiveCBF_STA_Controller
from modern_control_analysis import DCMotorStateSpace
from nonlinear_friction_smc_benchmark import DisturbanceObserverISMC, NonlinearDCMotor
from super_twisting_smc_eso import SuperTwistingSMC_ESO_Engine


def test_stribeck_friction_is_odd_and_finite():
    motor = NonlinearDCMotor()
    assert motor.stribeck_friction(0.0) == pytest.approx(0.0, abs=1e-9)
    for omega in (-2.0, -0.1, 0.1, 2.0):
        assert np.isfinite(motor.stribeck_friction(omega))
    assert motor.stribeck_friction(0.5) == pytest.approx(-motor.stribeck_friction(-0.5), rel=1e-9)
    assert motor.stribeck_friction(1.0) > 0.0
    assert motor.stribeck_friction(-1.0) < 0.0


def test_ismc_control_is_clipped_and_deterministic():
    motor = NonlinearDCMotor()
    controller = DisturbanceObserverISMC(motor)
    first = controller.compute_control(0.0, omega_ref=10.0, d_omega_ref=0.0, e_int=0.0, d_hat=0.0)
    second = controller.compute_control(0.0, omega_ref=10.0, d_omega_ref=0.0, e_int=0.0, d_hat=0.0)
    u_volts, s = first
    assert np.isfinite(u_volts) and np.isfinite(s)
    assert abs(u_volts) <= 24.0 + 1e-9
    assert first == second


def test_ndob_estimates_stay_bounded_under_updates():
    motor = NonlinearDCMotor()
    controller = DisturbanceObserverISMC(motor)
    for _ in range(200):
        estimate = controller.update_ndob(omega=0.0, u_volts=5.0, dt=0.001)
        assert np.isfinite(estimate)


def test_super_twisting_simulation_shapes_and_finiteness():
    engine = SuperTwistingSMC_ESO_Engine()
    t, w_ref, _t_dist, w_smc1, u_smc1, w_sta, u_sta, z2_eso = engine.simulate(
        t_end=0.05, dt=0.0005
    )
    n = len(t)
    assert n == len(w_ref) == len(w_smc1) == len(u_smc1) == len(w_sta) == len(u_sta) == len(z2_eso)
    assert t[0] == 0.0
    assert abs(float(t[-1]) - 0.05) <= 2 * 0.0005
    for series in (w_smc1, u_smc1, w_sta, u_sta, z2_eso):
        assert np.all(np.isfinite(series))


def test_adaptive_cbf_controller_voltage_limit():
    controller = AdaptiveCBF_STA_Controller()
    assert controller.V_max == 48.0


def test_state_space_is_controllable_observable_and_lqr_stabilizes():
    model = DCMotorStateSpace()
    rank_ctrl, rank_obs = model.check_controllability_observability()
    assert rank_ctrl == 2
    assert rank_obs == 2

    k_lqr, n_bar, p_solution = model.design_lqr()
    assert k_lqr.shape == (1, 2)
    assert np.isfinite(n_bar)
    assert p_solution.shape == (2, 2)

    eigenvalues = np.linalg.eigvals(model.A - model.B_mat @ k_lqr)
    assert np.all(eigenvalues.real < 0.0)
