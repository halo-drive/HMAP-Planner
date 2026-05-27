"""
HMAP Controller — kinematic trajectory controller for the HALO Planner.

Drop-in replacement for `waypoints_to_action()` in scripts/test_metadrive.py.
Consumes the planner's exact output contract (40 waypoints [x, y, heading, vel]
in ego frame + meta-action logits) and emits a MetaDrive action [steering, throttle]
in [-1, 1] — the same normalised shape as the Sygnal DBW percentage interface,
so behaviour validated here transfers to the on-vehicle HMAP Controller wrapper.

Design (matches the agreed structure):
    PlannerSanityChecker — Concern A: validates planner output each replan.
                       ok / clamp (feasibility) / reject (structural). Reject
                       coasts on last good plan, then brakes to stop if the
                       grace window expires.
    MotionModel      — kinematic bicycle, physical units. Used by the lateral
                       tracker and for an optional forward-rollout divergence check.
    TrajectoryTracker— pure-pursuit lateral (+ heading feedforward) and a
                       longitudinal PID that feedforwards the planner velocity.
    MetaActionGate   — softmax-weighted gain blend (comfort) + hard safety guards
                       (stop -> monotonic decel clamp; reverse -> forward lockout).
    HMAPController    — facade. compute_action(waypoints, meta_logits, agent).

Planner output contract (NOT modified — fixed upstream):
    waypoints:   (40, 4) float, ego frame, x-forward / y-left, heading rad, vel m/s
    meta_logits: (6,) float — [follow_lane, lane_change_left, lane_change_right,
                                stop, yield, reverse]

Coordinate frame: ego frame, x forward, y left (confirmed from world_to_ego and the
existing pure-pursuit arctan2(y, x) in the harness).
"""

from __future__ import annotations

import math
import numpy as np


# Meta-action index map — order matches model.py META_ACTIONS exactly.
META_FOLLOW_LANE = 0
META_LANE_CHANGE_LEFT = 1
META_LANE_CHANGE_RIGHT = 2
META_STOP = 3
META_YIELD = 4
META_REVERSE = 5
NUM_META = 6

DT_WAYPOINT = 0.1  # planner waypoint spacing (10 Hz / 4.0 s horizon)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# PlannerSanityChecker — Concern A: validate planner output before it drives.
# ---------------------------------------------------------------------------
# Verdicts:
#   "ok"     — plan is usable as-is.
#   "clamp"  — plan had violations that were repaired in place (monotonicity,
#              teleport jumps, velocity, curvature); keep driving.
#   "reject" — non-finite (NaN/inf) values only. Cannot be clamped, just
#              fabricated, and usually means planner inference broke. Caller
#              coasts on last good plan, then brakes if grace window expires.
#
# Policy (per design decision): clamp everything that CAN be clamped and keep
# driving; reject only the un-clampable NaN/inf case. meta-vs-geometry
# disagreement is NOT a sanity failure — it is handled softly by the gate, with
# only the flagrant reverse-vs-forward sign error reaching the reverse guard.
# ---------------------------------------------------------------------------
class PlannerSanityChecker:
    def __init__(self,
                 wheelbase: float,
                 max_delta_rad: float,
                 v_max: float = 30.0,
                 max_step_jump: float = 5.0):
        self.L = float(wheelbase)
        self.max_delta = float(max_delta_rad)
        self.v_max = float(v_max)
        # Max plausible distance between consecutive waypoints:
        # v_max * DT_WAYPOINT with margin. Catches teleport/garbage rows.
        self.max_step_jump = float(max_step_jump)

    def check(self, waypoints: np.ndarray) -> tuple:
        """
        Returns (verdict, clamped_waypoints, reason).
        clamped_waypoints is a (possibly modified) copy when verdict == 'clamp',
        else the original array (verdict 'ok') or None (verdict 'reject').
        """
        wp = waypoints

        # --- structural: NaN / inf -> REJECT (the only un-clampable case) ---
        # A non-finite value cannot be clamped, only fabricated, and usually
        # signals the planner inference itself broke. Holding the last good plan
        # is safer than driving on invented waypoints.
        if not np.all(np.isfinite(wp)):
            return ("reject", None, "non-finite values")

        # From here, every violation is repaired in-place and we keep driving.
        wp = wp.copy()
        clamped = False
        reasons = []

        x = wp[:, 0]
        y = wp[:, 1]

        # --- structural (repairable): enforce weakly-increasing forward x ---
        # Ego-frame x is forward; a backward regression is corruption, but we
        # repair rather than reject — clamp each x to be >= previous.
        dx = np.diff(x)
        if np.any(dx < -0.5):
            for i in range(1, len(x)):
                if x[i] < x[i - 1]:
                    x[i] = x[i - 1]
            wp[:, 0] = x
            clamped = True
            reasons.append("monotonicity")

        # --- structural (repairable): clip teleport jumps to plausible step ---
        steps = np.hypot(np.diff(wp[:, 0]), np.diff(y))
        if np.any(steps > self.max_step_jump):
            for i in range(1, len(wp)):
                seg = math.hypot(wp[i, 0] - wp[i - 1, 0], wp[i, 1] - wp[i - 1, 1])
                if seg > self.max_step_jump:
                    scale = self.max_step_jump / seg
                    wp[i, 0] = wp[i - 1, 0] + (wp[i, 0] - wp[i - 1, 0]) * scale
                    wp[i, 1] = wp[i - 1, 1] + (wp[i, 1] - wp[i - 1, 1]) * scale
            clamped = True
            reasons.append("jump")

        # refresh local views after structural repair
        x = wp[:, 0]
        y = wp[:, 1]

        # --- feasibility: velocity within [0, v_max] ---
        v = wp[:, 3]
        if np.any(v < 0) or np.any(v > self.v_max):
            wp[:, 3] = np.clip(v, 0.0, self.v_max)
            clamped = True
            reasons.append("velocity")

        # --- feasibility: curvature implies delta <= max_delta ---
        # Discrete curvature from heading change per arc length:
        #   kappa = dtheta / ds ; delta = atan(L * kappa)
        # Clamp the implied curvature by limiting heading rate of change.
        theta = wp[:, 2]
        ds = np.hypot(np.diff(x), np.diff(y))
        ds = np.clip(ds, 1e-3, None)
        dtheta = np.array([_wrap_angle(d) for d in np.diff(theta)])
        kappa = dtheta / ds
        max_kappa = math.tan(self.max_delta) / self.L
        if np.any(np.abs(kappa) > max_kappa):
            # Clamp heading channel so implied curvature is feasible.
            kappa_c = np.clip(kappa, -max_kappa, max_kappa)
            theta_c = theta.copy()
            for i in range(1, len(theta_c)):
                theta_c[i] = theta_c[i - 1] + kappa_c[i - 1] * ds[i - 1]
            wp[:, 2] = theta_c
            clamped = True
            reasons.append("curvature")

        if clamped:
            return ("clamp", wp, "+".join(reasons))
        return ("ok", waypoints, "")


# ---------------------------------------------------------------------------
# MotionModel — kinematic bicycle (physical units, vehicle-agnostic)
# ---------------------------------------------------------------------------
class MotionModel:
    """
    Kinematic bicycle. State [x, y, theta, v], controls [delta, a].

        x_dot     = v cos(theta)
        y_dot     = v sin(theta)
        theta_dot = (v / L) tan(delta)
        v_dot     = a

    Two roles:
      1. invert() — solve road-wheel angle delta to reach a lookahead point
         (this is what the lateral tracker uses).
      2. rollout() — forward-simulate under a control plan for the optional
         divergence check (Concern B). Off by default in this first pass.
    """

    def __init__(self, wheelbase: float, max_delta_rad: float):
        self.L = float(wheelbase)
        self.max_delta = float(max_delta_rad)

    def pure_pursuit_delta(self, target_x: float, target_y: float) -> float:
        """
        Road-wheel angle to steer toward an ego-frame lookahead point.
        Standard pure pursuit: delta = atan2(2 L sin(alpha), ld).
        """
        ld = max(math.hypot(target_x, target_y), 1e-3)
        alpha = math.atan2(target_y, target_x)  # bearing to target in ego frame
        delta = math.atan2(2.0 * self.L * math.sin(alpha), ld)
        return float(np.clip(delta, -self.max_delta, self.max_delta))

    def step(self, state: np.ndarray, delta: float, a: float, dt: float) -> np.ndarray:
        """One forward Euler step. state = [x, y, theta, v]."""
        x, y, theta, v = state
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += (v / self.L) * math.tan(delta) * dt
        v += a * dt
        return np.array([x, y, theta, max(v, 0.0)], dtype=np.float64)

    def rollout(self, delta: float, a: float, v0: float, horizon_s: float,
                dt: float = 0.02) -> np.ndarray:
        """
        Forward-simulate from ego origin under constant (delta, a).
        Returns array of states. Used only for the optional divergence check.
        """
        state = np.array([0.0, 0.0, 0.0, v0], dtype=np.float64)
        n = int(horizon_s / dt)
        out = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            state = self.step(state, delta, a, dt)
            out[i] = state
        return out


# ---------------------------------------------------------------------------
# MetaActionGate — blend comfort gains, enforce hard safety guards
# ---------------------------------------------------------------------------
class MetaActionGate:
    """
    Turns meta-action probabilities into:
      - a smooth lookahead/comfort schedule (blended by probability), and
      - hard guards that override the velocity target (gated by threshold).

    Blending is for *tuning* (lookahead distance). Guards are for *safety*
    (stop clamp, reverse lockout) and are thresholded, never averaged — a
    30%-probability stop must not become a soft slowdown.
    """

    # Per-meta-action nominal lookahead index into the 40-waypoint array.
    # Larger index = look further ahead = smoother/faster; smaller = tighter.
    LOOKAHEAD_IDX = {
        META_FOLLOW_LANE: 10,
        META_LANE_CHANGE_LEFT: 14,
        META_LANE_CHANGE_RIGHT: 14,
        META_STOP: 5,
        META_YIELD: 7,
        META_REVERSE: 5,
    }

    # Probability above which a guard activates.
    STOP_THRESHOLD = 0.35
    YIELD_THRESHOLD = 0.40
    REVERSE_THRESHOLD = 0.50

    def __init__(self):
        self._idx_vec = np.array(
            [self.LOOKAHEAD_IDX[i] for i in range(NUM_META)], dtype=np.float64
        )

    def schedule(self, probs: np.ndarray) -> dict:
        """
        Returns a control schedule:
            lookahead_idx : float (blended; caller rounds/clamps)
            stop          : bool  (hard guard)
            yield_active  : bool  (hard guard, softer than stop)
            reverse       : bool  (hard guard)
        """
        lookahead_idx = float(np.dot(probs, self._idx_vec))

        stop = probs[META_STOP] >= self.STOP_THRESHOLD
        yield_active = probs[META_YIELD] >= self.YIELD_THRESHOLD
        reverse = probs[META_REVERSE] >= self.REVERSE_THRESHOLD

        return {
            "lookahead_idx": lookahead_idx,
            "stop": stop,
            "yield_active": yield_active,
            "reverse": reverse,
        }


# ---------------------------------------------------------------------------
# TrajectoryTracker — lateral (pure pursuit + heading FF) + longitudinal PID
# ---------------------------------------------------------------------------
class TrajectoryTracker:
    """
    Closed-loop tracker. Runs every sim tick against live measured ego speed.

    Lateral:  pure-pursuit position tracking to a meta-scheduled lookahead point,
              plus a small heading feedforward from the planner's theta channel.
    Longitudinal: PID around a target speed that *feedforwards* the planner's
              velocity at the lookahead waypoint, clamped to a geometrically
              plausible band. Guards from the gate override the target.
    """

    def __init__(self, motion: MotionModel,
                 kp: float = 0.6, ki: float = 0.05, kd: float = 0.1,
                 heading_ff_gain: float = 0.15,
                 max_decel_target: float = 6.0):
        self.motion = motion
        self.kp, self.ki, self.kd = kp, ki, kd
        self.heading_ff_gain = heading_ff_gain
        self.max_decel_target = max_decel_target

        # PID state (persists across ticks)
        self._int = 0.0
        self._prev_err = 0.0

    def reset(self):
        self._int = 0.0
        self._prev_err = 0.0

    # ---- lateral ----------------------------------------------------------
    def _steering(self, waypoints: np.ndarray, idx: int, max_delta: float) -> float:
        tx, ty = float(waypoints[idx, 0]), float(waypoints[idx, 1])
        delta = self.motion.pure_pursuit_delta(tx, ty)

        # Heading feedforward: the planner's theta at the lookahead point is the
        # path tangent. Nudge delta toward it — captures curvature that pure
        # position pursuit lags on. Kept small so position tracking dominates.
        path_theta = float(waypoints[idx, 2])
        delta = delta + self.heading_ff_gain * _wrap_angle(path_theta)
        return float(np.clip(delta, -max_delta, max_delta))

    # ---- longitudinal -----------------------------------------------------
    def _target_speed(self, waypoints: np.ndarray, idx: int, sched: dict) -> float:
        # Feedforward: trust the planner's velocity channel at the lookahead wp.
        v_plan = float(waypoints[idx, 3])

        # Geometric sanity band: speed implied by how far the far waypoint is.
        far_idx = min(39, len(waypoints) - 1)
        far_dist = math.hypot(waypoints[far_idx, 0], waypoints[far_idx, 1])
        v_geom = far_dist / ((far_idx + 1) * DT_WAYPOINT)
        # Clamp planner v to [0, 1.5x geometric] — rejects a spuriously high v.
        v_target = float(np.clip(v_plan, 0.0, 1.5 * max(v_geom, 0.1)))

        # --- hard guards override ---
        if sched["stop"]:
            v_target = 0.0
        elif sched["yield_active"]:
            v_target = min(v_target, 0.3 * v_geom)  # heavy slow, not full stop
        return v_target

    def _throttle(self, v_target: float, v_meas: float, dt: float) -> float:
        err = v_target - v_meas
        self._int += err * dt
        self._int = float(np.clip(self._int, -5.0, 5.0))  # anti-windup
        deriv = (err - self._prev_err) / max(dt, 1e-3)
        self._prev_err = err

        u = self.kp * err + self.ki * self._int + self.kd * deriv
        # Map PID output to MetaDrive's single throttle/brake axis [-1, 1].
        return float(np.clip(u, -1.0, 1.0))

    # ---- combined ---------------------------------------------------------
    def compute(self, waypoints: np.ndarray, sched: dict,
                v_meas: float, max_delta: float, dt: float) -> list:
        n = len(waypoints)
        idx = int(np.clip(round(sched["lookahead_idx"]), 1, n - 1))

        # Reverse guard: lock out forward motion, hold straight, command brake.
        # (Forward waypoints + reverse meta = contradiction -> safe hold.)
        if sched["reverse"]:
            self.reset()
            return [0.0, -0.5]

        steering_rad = self._steering(waypoints, idx, max_delta)
        steering_norm = float(np.clip(steering_rad / max_delta, -1.0, 1.0))

        v_target = self._target_speed(waypoints, idx, sched)
        throttle = self._throttle(v_target, v_meas, dt)

        return [steering_norm, throttle]


# ---------------------------------------------------------------------------
# HMAPController — facade. One-line swap for waypoints_to_action().
# ---------------------------------------------------------------------------
class HMAPController:
    """
    Usage in test_metadrive.py:

        from hmap_controller import HMAPController
        controller = HMAPController(env.agent)          # once, after reset
        ...
        action = controller.compute_action(
            current_waypoints, current_meta_logits, env.agent, dt=0.025
        )
        env.step(action)

    dt is the sim tick period: MetaDrive default physics is 0.02s, but the
    harness runs decision steps at 40 Hz -> dt = 0.025. Pass the value matching
    your env; it only affects the PID derivative/integral scaling.
    """

    def __init__(self, agent, enable_rollout_check: bool = False,
                 grace_ticks: int = 8, v_max: float = 30.0,
                 kp: float = 0.2, ki: float = 0.05, kd: float = 0.1):
        wheelbase = agent.FRONT_WHEELBASE + agent.REAR_WHEELBASE
        max_delta = math.radians(agent.max_steering)
        self.motion = MotionModel(wheelbase, max_delta)
        self.gate = MetaActionGate()
        self.tracker = TrajectoryTracker(self.motion, kp=kp, ki=ki, kd=kd)
        self.sanity = PlannerSanityChecker(wheelbase, max_delta, v_max=v_max)
        self.max_delta = max_delta
        self.enable_rollout_check = enable_rollout_check
        self._last_divergence = 0.0

        # Tiered-fallback state (Concern A).
        self.grace_ticks = int(grace_ticks)
        self._last_good_wp = None        # last validated/clamped waypoints
        self._reject_streak = 0          # consecutive rejects
        self._last_verdict = "ok"
        self._last_reason = ""

    def reset(self):
        self.tracker.reset()
        self._last_good_wp = None
        self._reject_streak = 0
        self._last_verdict = "ok"
        self._last_reason = ""

    def compute_action(self, waypoints, meta_logits, agent, dt: float = 0.025) -> list:
        waypoints = np.asarray(waypoints, dtype=np.float64)
        probs = _softmax(np.asarray(meta_logits, dtype=np.float64).ravel())
        sched = self.gate.schedule(probs)
        v_meas = float(np.linalg.norm(agent.velocity))

        # --- Concern A: validate planner output before it drives anything ---
        verdict, checked_wp, reason = self.sanity.check(waypoints)
        self._last_verdict, self._last_reason = verdict, reason

        if verdict == "reject":
            self._reject_streak += 1
            if self._last_good_wp is not None and self._reject_streak <= self.grace_ticks:
                # Coast on last good plan during the grace window.
                active_wp = self._last_good_wp
            else:
                # Grace expired (or never had a good plan): controlled brake-to-stop.
                self.tracker.reset()
                brake = -0.6 if v_meas > 0.2 else 0.0
                return [0.0, brake]
        else:
            # ok or clamp: this becomes the new last-good plan.
            active_wp = checked_wp
            self._last_good_wp = active_wp
            self._reject_streak = 0

        action = self.tracker.compute(active_wp, sched, v_meas, self.max_delta, dt)

        if self.enable_rollout_check:
            self._last_divergence = self._rollout_divergence(active_wp, action, v_meas)

        return action

    @property
    def last_verdict(self) -> tuple:
        """(verdict, reason, reject_streak) — for harness logging/metrics."""
        return (self._last_verdict, self._last_reason, self._reject_streak)

    def _rollout_divergence(self, waypoints, action, v_meas, horizon_s: float = 1.0) -> float:
        """Mean lateral gap between predicted path and reference over horizon_s."""
        steering_norm, throttle = action
        delta = steering_norm * self.max_delta
        a = throttle * 3.0  # rough accel scale; calibrate later
        pred = self.motion.rollout(delta, a, v_meas, horizon_s)
        # Compare predicted (x,y) to nearest reference waypoint (cheap proxy).
        ref = waypoints[:, :2]
        gaps = []
        for p in pred[::5]:
            d = np.linalg.norm(ref - p[:2], axis=1).min()
            gaps.append(d)
        return float(np.mean(gaps)) if gaps else 0.0

    @property
    def last_divergence(self) -> float:
        return self._last_divergence
