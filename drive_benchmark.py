#!/usr/bin/env python3
"""Shared benchmark harness for the precision DC drive study.

One plant, one scenario, one metric definition set, used by every controller
comparison in this repository. Every reported number is produced by this
module and written to results/precision_drive_benchmark.json.

Plant (SI units, two states: armature current i, rotor speed w):

    di/dt = (V - R*i - Ke*w) / L
    dw/dt = (Kt*i - b*w - T_load) / J

Parameters are representative values for a 100 W-class brushed servo motor
with a reflected load (documented, not copied from a vendor datasheet):

    J  = 5.0e-4 kg m^2     rotor + reflected load inertia
    b  = 5.0e-5 N m s/rad  viscous friction
    Kt = 0.05  N m/A       torque constant
    Ke = 0.05  V s/rad     back-EMF constant
    R  = 1.0   ohm         armature resistance
    L  = 1.0e-3 H          armature inductance
    V_max = 48 V           DC bus limit
    I_max = 20 A           drive current limit

At the 100 rad/s operating point the back-EMF is 5 V and the 0.45 N m load
needs 9.1 A (14.1 V), so the operating point is reachable with margin. The
earlier harness used Kt/J with volts substituted for amps and a bus limit of
48 V against a plant whose maximum speed was 4.8 rad/s; this module replaces
it and every legacy script delegates here.

Controller interface: every speed controller commands an armature current
i_ref in amps; a shared model-based voltage stage maps it to the bus:

    V = R * i_ref + Ke * w,  then clipped to +/- V_max

The adaptive controller's CBF filter acts on i_ref through the closed-form
solution of the linear voltage-constraint QP (the barrier is state dependent:
the admissible current shrinks as speed rises), which is a genuine
state-constraint filter, not a constant clip.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Plant and scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DCMotorParams:
    J: float = 5.0e-4
    b: float = 5.0e-5
    Kt: float = 0.05
    Ke: float = 0.05
    R: float = 1.0
    L: float = 1.0e-3
    V_max: float = 48.0
    I_max: float = 20.0

    def max_speed_no_load(self) -> float:
        """Steady-state speed at V_max with no load (rad/s)."""
        # w_ss solves V = Ke*w + R*(b*w)/Kt
        return self.V_max / (self.Ke + self.R * self.b / self.Kt)


@dataclass(frozen=True)
class Scenario:
    t_end: float = 3.0
    dt: float = 1.0e-4
    w_ref_value: float = 100.0
    ref_step_at: float = 0.2
    load_step_at: float = 1.2
    load_step_nm: float = 0.45
    load_ripple_nm: float = 0.15
    load_ripple_hz: float = 20.0
    stribeck_tc_nm: float = 0.0
    stribeck_ts_nm: float = 0.0
    stribeck_ws_rad_s: float = 5.0


def stribeck_torque(w: float, sc: Scenario) -> float:
    if sc.stribeck_tc_nm <= 0.0 and sc.stribeck_ts_nm <= 0.0:
        return 0.0
    return (sc.stribeck_tc_nm + (sc.stribeck_ts_nm - sc.stribeck_tc_nm)
            * np.exp(-((w / sc.stribeck_ws_rad_s) ** 2))) * np.tanh(10.0 * w)


def reference(t: float, sc: Scenario) -> float:
    return sc.w_ref_value if t >= sc.ref_step_at else 0.0


def load_torque(t: float, sc: Scenario) -> float:
    if t < sc.load_step_at:
        return 0.0
    return sc.load_step_nm + sc.load_ripple_nm * np.sin(
        2.0 * np.pi * sc.load_ripple_hz * (t - sc.load_step_at)
    )


def _derivatives(i: float, w: float, V: float, T_load: float, p: DCMotorParams) -> tuple[float, float]:
    di = (V - p.R * i - p.Ke * w) / p.L
    dw = (p.Kt * i - p.b * w - T_load) / p.J
    return di, dw


# ---------------------------------------------------------------------------
# Controllers (each commands i_ref in amps)
# ---------------------------------------------------------------------------

class PIDController:
    """Velocity-form PID with clamped integral (anti-windup)."""

    def __init__(self, kp: float, ki: float, kd: float, i_max: float) -> None:
        self.kp, self.ki, self.kd, self.i_max = kp, ki, kd, i_max
        self._integral = 0.0
        self._prev_e: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_e = None

    def step(self, e: float, dt: float) -> float:
        de = 0.0 if self._prev_e is None else (e - self._prev_e) / dt
        self._prev_e = e
        raw = self.kp * e + self.ki * self._integral + self.kd * de
        i_ref = float(np.clip(raw, -self.i_max, self.i_max))
        if abs(raw) < self.i_max or (raw * e < 0.0):  # integrate unless pushing further into saturation
            self._integral += e * dt
        return i_ref


class FirstOrderSMC:
    """First-order sliding mode in current-command space with a boundary layer."""

    def __init__(self, k_sw: float, phi: float, friction_ff: bool = True, params: DCMotorParams | None = None) -> None:
        self.k_sw, self.phi = k_sw, phi
        self._params = params

    def reset(self) -> None:
        pass

    def step(self, e: float, w: float, dt: float) -> float:
        p = self._params
        ff = (p.b * w / p.Kt) if (p is not None) else 0.0
        return ff + self.k_sw * float(np.tanh(e / self.phi))


class SuperTwistingESO:
    """Super-twisting 2-SMC with a linear extended state observer (speed only).

    Mechanical model used by the observer:  dw/dt = (Kt*i - b*w)/J - d,
    with d the lumped load acceleration estimated by z2. The observer is
    driven by the commanded current (available sensorless in the drive).
    """

    def __init__(
        self,
        k1: float,
        k2: float,
        omega_o: float,
        params: DCMotorParams,
        adaptive: bool = False,
        adaptive_gamma: float = 3000.0,
        adaptive_decay: float = 5.0,
        adaptive_eps: float = 0.05,
        k1_min: float = 100.0,
        k1_max: float = 1600.0,
    ) -> None:
        self.k1 = k1
        self.k2 = k2
        self.omega_o = omega_o
        self._params = params
        self.adaptive = adaptive
        self.gamma = adaptive_gamma
        self.decay = adaptive_decay
        self.eps = adaptive_eps
        self.k1_min = k1_min
        self.k1_max = k1_max
        self.reset()

    def reset(self) -> None:
        self.z1 = 0.0
        self.z2 = 0.0
        self.v_int = 0.0
        self.k1_curr = self.k1_min if self.adaptive else self.k1
        self.k1_hist: list[float] = []
        self.z2_hist: list[float] = []

    def step(self, e: float, w: float, i_cmd_prev: float, dt: float) -> float:
        p = self._params
        beta1, beta2 = 2.0 * self.omega_o, self.omega_o ** 2

        # Observer update (Euler at the harness dt; fast relative to the plant)
        err = self.z1 - w
        dz1 = self.z2 - beta1 * err + (p.Kt * i_cmd_prev - p.b * w) / p.J
        dz2 = -beta2 * err
        self.z1 += dz1 * dt
        self.z2 += dz2 * dt
        self.z2_hist.append(self.z2)

        if self.adaptive:
            # Fast gain increase while the sliding variable is outside the dead
            # zone; smooth relaxation back to the base gain in steady state.
            if abs(e) > self.eps:
                self.k1_curr = float(np.clip(
                    self.k1_curr + self.gamma * abs(e) * dt, self.k1_min, self.k1_max
                ))
            else:
                self.k1_curr = float(np.clip(
                    self.k1_curr - self.decay * (self.k1_curr - self.k1_min) * dt,
                    self.k1_min, self.k1_max,
                ))
            k2 = 2.5 * self.k1_curr
            k1 = self.k1_curr
        else:
            k1, k2 = self.k1, self.k2
        self.k1_hist.append(self.k1_curr)

        # Super-twisting core with conditional integration (anti-windup against
        # the actuator current limit). The observer's z2 converges to MINUS the
        # lumped load acceleration (the original convention), hence the minus.
        step_int = k2 * np.sign(e) * dt
        self.v_int += step_int
        v_core = k1 * np.sqrt(abs(e)) * np.sign(e) + self.v_int
        i_ref = (p.b * w / p.Kt) - self.z2 * p.J / p.Kt + v_core * p.J / p.Kt
        if abs(i_ref) > p.I_max and np.sign(i_ref) == np.sign(e):
            self.v_int -= step_int  # do not wind up while saturated
            v_core = k1 * np.sqrt(abs(e)) * np.sign(e) + self.v_int
            i_ref = (p.b * w / p.Kt) - self.z2 * p.J / p.Kt + v_core * p.J / p.Kt
        self.v_int = float(np.clip(self.v_int, -5.0e4, 5.0e4))
        return float(i_ref)


class VoltageBarrierFilter:
    """Closed-form QP for the linear voltage constraint |R i + Ke w| <= V_max.

    The admissible current set is state dependent (it shrinks as speed rises),
    so this is a genuine control-barrier filter rather than an output clip.
    """

    def __init__(self, params: DCMotorParams) -> None:
        self._params = params

    def admissible_current(self, w: float) -> float:
        p = self._params
        return max(0.0, (p.V_max - p.Ke * max(0.0, w)) / p.R)

    def filter(self, i_ref: float, w: float) -> float:
        """Project i_ref onto the admissible set (voltage barrier AND drive limit)."""
        i_adm = min(self.admissible_current(w), self._params.I_max)
        return float(np.clip(i_ref, -i_adm, i_adm))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_controller(
    controller,
    params: DCMotorParams,
    scenario: Scenario,
    use_barrier: bool = False,
) -> dict[str, np.ndarray]:
    """RK4 integration of the plant under one current-command controller."""
    n = int(round(scenario.t_end / scenario.dt)) + 1
    t = np.arange(n) * scenario.dt
    w = np.zeros(n)
    i_state = np.zeros(n)
    i_ref_hist = np.zeros(n)
    V_hist = np.zeros(n)
    w_ref_hist = np.zeros(n)
    T_load_hist = np.zeros(n)

    barrier = VoltageBarrierFilter(params) if use_barrier else None
    controller.reset()
    i_ref_prev = 0.0

    for k in range(n - 1):
        tk = t[k]
        w_ref = reference(tk, scenario)
        T_L = load_torque(tk, scenario) + stribeck_torque(w[k], scenario)
        e = w_ref - w[k]

        if isinstance(controller, SuperTwistingESO):
            i_ref = controller.step(e, w[k], i_ref_prev, scenario.dt)
        elif isinstance(controller, FirstOrderSMC):
            i_ref = controller.step(e, w[k], scenario.dt)
        else:
            i_ref = controller.step(e, scenario.dt)

        if barrier is not None:
            i_ref = barrier.filter(i_ref, w[k])
        else:
            i_ref = float(np.clip(i_ref, -params.I_max, params.I_max))

        V = float(np.clip(params.R * i_ref + params.Ke * w[k], -params.V_max, params.V_max))

        # RK4 on the two-state plant
        i0, w0 = i_state[k], w[k]
        di1, dw1 = _derivatives(i0, w0, V, T_L, params)
        di2, dw2 = _derivatives(i0 + 0.5 * scenario.dt * di1, w0 + 0.5 * scenario.dt * dw1, V, T_L, params)
        di3, dw3 = _derivatives(i0 + 0.5 * scenario.dt * di2, w0 + 0.5 * scenario.dt * dw2, V, T_L, params)
        di4, dw4 = _derivatives(i0 + scenario.dt * di3, w0 + scenario.dt * dw3, V, T_L, params)
        i_state[k + 1] = i0 + (scenario.dt / 6.0) * (di1 + 2 * di2 + 2 * di3 + di4)
        w[k + 1] = w0 + (scenario.dt / 6.0) * (dw1 + 2 * dw2 + 2 * dw3 + dw4)

        i_ref_hist[k] = i_ref
        V_hist[k] = V
        w_ref_hist[k] = w_ref
        T_load_hist[k] = T_L
        i_ref_prev = i_ref

    for arr in (i_ref_hist, V_hist, w_ref_hist, T_load_hist):
        arr[-1] = arr[-2]

    out = {
        "t": t,
        "w": w,
        "i": i_state,
        "i_ref": i_ref_hist,
        "V": V_hist,
        "w_ref": w_ref_hist,
        "T_load": T_load_hist,
    }
    if isinstance(controller, SuperTwistingESO):
        z2_series = controller.z2_hist + [controller.z2_hist[-1]] if controller.z2_hist else np.zeros(n)
        k1_series = controller.k1_hist + [controller.k1_hist[-1]] if controller.k1_hist else np.zeros(n)
        out["z2"] = np.asarray(z2_series)
        out["k1_hist"] = np.asarray(k1_series)
    return out


# ---------------------------------------------------------------------------
# Metrics (single fixed definition set)
# ---------------------------------------------------------------------------

def compute_metrics(traj: dict[str, np.ndarray], sc: Scenario) -> dict[str, float]:
    t = traj["t"]
    w = traj["w"]
    V = traj["V"]
    ref = sc.w_ref_value

    settle_mask = (t >= sc.ref_step_at) & (t < sc.load_step_at)
    pre_mask = (t >= 0.8) & (t < sc.load_step_at)
    post_mask = t >= sc.load_step_at

    # Speed sag: steady pre-disturbance mean minus the post-disturbance minimum
    sag = float(np.mean(w[pre_mask]) - np.min(w[post_mask]))

    # Recovery: first time after the load step within 2% of the reference, held 0.1 s
    tol = 0.02 * ref
    recovery = float("nan")
    hold = int(0.1 / sc.dt)
    idx = np.where(post_mask)[0]
    for j in idx:
        if j + hold < len(t) and np.all(np.abs(w[j:j + hold] - ref) <= tol):
            recovery = float(t[j] - sc.load_step_at)
            break

    # Settling after the reference step (2% band, held 0.1 s)
    settling = float("nan")
    idx2 = np.where((t >= sc.ref_step_at))[0]
    for j in idx2:
        if j + hold < len(t) and np.all(np.abs(w[j:j + hold] - ref) <= tol):
            settling = float(t[j] - sc.ref_step_at)
            break

    overshoot = float(max(0.0, (np.max(w[settle_mask]) - ref) / ref * 100.0)) if settle_mask.any() else 0.0
    rms_err = float(np.sqrt(np.mean((w[t >= 0.3] - ref) ** 2)))

    # Steady-state chattering: RMS of dV/dt over the final window after all
    # transient responses have decayed (V/s).
    ss_start = max(sc.load_step_at + 0.3, t[-1] - 1.0)
    ss = t >= ss_start
    if not ss.any():
        ss = t >= 0.9 * t[-1]
    dv = np.diff(V[ss]) / sc.dt
    chattering = float(np.sqrt(np.mean(dv ** 2))) if dv.size else float("nan")

    # Transient control effort: RMS of dV/dt across the load-step response
    tr = (t >= sc.load_step_at) & (t < sc.load_step_at + 0.5)
    dv_tr = np.diff(V[tr]) / sc.dt
    chattering_transient = float(np.sqrt(np.mean(dv_tr ** 2)))

    return {
        "sag_rad_s": sag,
        "recovery_time_s": recovery,
        "settling_time_s": settling,
        "overshoot_pct": overshoot,
        "rms_tracking_error_rad_s": rms_err,
        "chattering_v_per_s": chattering,
        "chattering_transient_v_per_s": chattering_transient,
        "peak_voltage_v": float(np.max(np.abs(V))),
        "peak_current_a": float(np.max(np.abs(traj["i"]))),
    }


# ---------------------------------------------------------------------------
# Tuning (one procedure per controller class, documented budgets)
# ---------------------------------------------------------------------------

TUNE_SCENARIO = Scenario(t_end=2.5, dt=2.0e-4)


def _score(m: dict[str, float]) -> float:
    """Tuning objective, identical for every controller class.

    RMS tracking error + speed sag, plus a chattering term at a documented
    trade-off weight: 1000 V/s of steady-state control chattering is treated
    as equivalent to 1 rad/s of tracking error. Without the chattering term a
    grid search degenerates to maximum aggressiveness and manufactures the
    classical chattering straw man.
    """
    return (
        m["rms_tracking_error_rad_s"]
        + m["sag_rad_s"]
        + 1.0e-3 * m["chattering_v_per_s"]
    )


def _tune_grid(params: DCMotorParams, sc: Scenario, factory, grid: list) -> object:
    best_score, best = float("inf"), grid[0]
    for gains in grid:
        traj = simulate_controller(factory(gains), params, sc)
        s = _score(compute_metrics(traj, sc))
        if s < best_score:
            best_score, best = s, gains
    return best


def tune_all(params: DCMotorParams, sc: Scenario | None = None, force: bool = False) -> dict[str, object]:
    """Coarse grid search per controller class on a shortened scenario.

    Budgets: PID 60 candidates (kp x ki x kd), first-order SMC 9 (k_sw x phi),
    super-twisting 40 (k1 x k2 x observer bandwidth), adaptive 9
    (adaptation rate x gain floor). Selection metric: RMS tracking error +
    speed sag + 1e-3 x steady-state chattering (see _score).

    Results are cached in results/tuned_gains.json keyed by plant+scenario so
    tests and repeated runs do not re-tune. Pass force=True to re-run the
    search (required after changing the plant or the scenario).
    """
    sc = sc or TUNE_SCENARIO
    cache = RESULTS_DIR / "tuned_gains.json"
    if not force and cache.exists():
        try:
            blob = json.loads(cache.read_text(encoding="utf-8"))
            if blob.get("plant") == asdict(params) and blob.get("scenario") == asdict(sc):
                return blob["gains"]
        except (json.JSONDecodeError, KeyError):
            pass

    pid = _tune_grid(
        params, sc,
        lambda g: PIDController(g[0], g[1], g[2], params.I_max),
        [(kp, ki, kd) for kp in (0.2, 0.5, 1.0, 2.0) for ki in (0.5, 1.0, 2.0, 5.0, 10.0) for kd in (0.0, 0.01, 0.05)],
    )
    smc = _tune_grid(
        params, sc,
        lambda g: FirstOrderSMC(g[0], g[1], params=params),
        [(k, phi) for k in (10.0, 12.0, 15.0) for phi in (0.5, 1.0, 2.0)],
    )
    sta = _tune_grid(
        params, sc,
        lambda g: SuperTwistingESO(g[0], g[1], g[2], params),
        [(k1, k2, wo) for k1 in (50.0, 100.0, 200.0, 400.0, 800.0) for k2 in (500.0, 1000.0, 3000.0, 10000.0) for wo in (60.0, 120.0)],
    )
    adaptive = _tune_grid(
        params, sc,
        lambda g: SuperTwistingESO(
            sta[0], sta[1], sta[2], params,
            adaptive=True, adaptive_gamma=g[0], k1_min=g[1], k1_max=1600.0,
        ),
        [(gamma, kmin) for gamma in (3000.0, 10000.0, 30000.0) for kmin in (100.0, 400.0, 800.0)],
    )
    gains = {
        "pid": {"kp": pid[0], "ki": pid[1], "kd": pid[2]},
        "smc1": {"k_sw": smc[0], "phi": smc[1]},
        "sta_eso": {"k1": sta[0], "k2": sta[1], "omega_o": sta[2]},
        "adaptive": {"gamma": adaptive[0], "k1_min": adaptive[1], "k1_max": 1600.0},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"plant": asdict(params), "scenario": asdict(sc), "gains": gains}, indent=2) + "\n",
        encoding="utf-8",
    )
    return gains


def controller_suite(
    params: DCMotorParams,
    gains: dict[str, dict],
) -> dict[str, tuple[object, bool]]:
    """All controllers under comparison: (controller, use_voltage_barrier).

    All four controllers are tuned by the same procedure (see tune_all) and
    operate in the same current-command space on the same plant.
    """
    g_pid, g_smc, g_sta = gains["pid"], gains["smc1"], gains["sta_eso"]
    return {
        "pid": (PIDController(g_pid["kp"], g_pid["ki"], g_pid["kd"], params.I_max), False),
        "smc1": (FirstOrderSMC(g_smc["k_sw"], g_smc["phi"], params=params), False),
        "sta_eso": (
            SuperTwistingESO(g_sta["k1"], g_sta["k2"], g_sta["omega_o"], params),
            False,
        ),
        "adaptive_sta_eso_cbf": (
            SuperTwistingESO(
                g_sta["k1"], g_sta["k2"], g_sta["omega_o"], params,
                adaptive=True,
                adaptive_gamma=gains["adaptive"]["gamma"],
                k1_min=gains["adaptive"]["k1_min"],
                k1_max=gains["adaptive"]["k1_max"],
            ),
            True,
        ),
    }


def run_full_benchmark(
    params: DCMotorParams | None = None,
    sc: Scenario | None = None,
    mc_samples: int = 20,
    mc_seed: int = 20261011,
) -> dict:
    """Run every controller on the shared scenario plus a Monte Carlo sweep."""
    params = params or DCMotorParams()
    sc = sc or Scenario()

    if params.max_speed_no_load() < sc.w_ref_value * 1.5:
        raise ValueError("plant cannot reach the reference with margin; fix parameters")

    gains = tune_all(params)
    suite = controller_suite(params, gains)
    results: dict[str, dict] = {}
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for name, (ctrl, use_barrier) in suite.items():
        traj = simulate_controller(ctrl, params, sc, use_barrier=use_barrier)
        trajectories[name] = traj
        results[name] = compute_metrics(traj, sc)

    # Monte Carlo: +-20% on J, b, Kt, R (fixed seed); nominal gains reused
    rng = np.random.default_rng(mc_seed)
    mc: dict[str, dict[str, list[float]]] = {name: {"sag_rad_s": [], "rms_tracking_error_rad_s": []} for name in suite}
    for _ in range(mc_samples):
        perturbed = DCMotorParams(
            J=float(params.J * rng.uniform(0.8, 1.2)),
            b=float(params.b * rng.uniform(0.8, 1.2)),
            Kt=float(params.Kt * rng.uniform(0.8, 1.2)),
            Ke=params.Ke,
            R=float(params.R * rng.uniform(0.8, 1.2)),
            L=params.L,
            V_max=params.V_max,
            I_max=params.I_max,
        )
        suite_mc = controller_suite(perturbed, gains)
        for name, (ctrl, use_barrier) in suite_mc.items():
            traj = simulate_controller(ctrl, perturbed, sc, use_barrier=use_barrier)
            m = compute_metrics(traj, sc)
            mc[name]["sag_rad_s"].append(m["sag_rad_s"])
            mc[name]["rms_tracking_error_rad_s"].append(m["rms_tracking_error_rad_s"])

    mc_summary = {
        name: {
            "sag_mean": float(np.mean(vals["sag_rad_s"])),
            "sag_std": float(np.std(vals["sag_rad_s"])),
            "rms_error_mean": float(np.mean(vals["rms_tracking_error_rad_s"])),
            "rms_error_std": float(np.std(vals["rms_tracking_error_rad_s"])),
        }
        for name, vals in mc.items()
    }

    return {
        "plant": asdict(params),
        "scenario": asdict(sc),
        "gains": gains,
        "controllers": results,
        "monte_carlo": {"samples": mc_samples, "seed": mc_seed, "summary": mc_summary},
    }, trajectories


def main() -> int:
    payload, trajectories = run_full_benchmark()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "precision_drive_benchmark.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print("PRECISION DRIVE DYNAMICS: SHARED HARNESS (single plant, single scenario)")
    print("=" * 78)
    g = payload["gains"]
    print(f"tuned gains: PID {g['pid']}  |  SMC1 {g['smc1']}  |  STA {g['sta_eso']}")
    header = f"{'controller':<24} {'sag(rad/s)':>10} {'recover(s)':>10} {'rms_err':>9} {'chatter(V/s)':>13} {'peakV':>6}"
    print(header)
    print("-" * len(header))
    for name, m in payload["controllers"].items():
        rec = m["recovery_time_s"]
        rec_text = f"{rec:>10.3f}" if rec == rec else f"{'no 2% recovery':>10}"
        print(
            f"{name:<24} {m['sag_rad_s']:>10.3f} {rec_text} "
            f"{m['rms_tracking_error_rad_s']:>9.3f} {m['chattering_v_per_s']:>13.0f} {m['peak_voltage_v']:>6.1f}"
        )
    print("-" * len(header))
    print("Monte Carlo (+-20% J,b,Kt,R):")
    for name, s in payload["monte_carlo"]["summary"].items():
        print(f"  {name:<24} sag {s['sag_mean']:.3f} +/- {s['sag_std']:.3f} rad/s   rms {s['rms_error_mean']:.3f} +/- {s['rms_error_std']:.3f}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
