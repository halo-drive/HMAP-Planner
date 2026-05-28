"""
HALO Planner — MetaDrive closed-loop evaluation.

Runs the trained planner in MetaDrive simulator:
  1. Reads ego/traffic/lane state from MetaDrive (replaces FusionEngine)
  2. Converts to planner input tensors (same format as FusedPacket)
  3. Runs inference through best_planner.pt
  4. Converts output waypoints to steering/throttle via the HMAP Controller
     (kinematic bicycle tracker + meta-action gate + planner-sanity stack)
  5. Steps the simulator — full closed loop

The planner output contract is NOT modified. All trajectory-to-control logic
lives in hmap_controller.HMAPController, which mirrors the on-vehicle HMAP
Controller wrapper that drives the Sygnal DBW interface. The legacy pure-pursuit
function waypoints_to_action() is retained only for --compare A/B baselining.

Usage:
    # 3D rendered window (needs display)
    python scripts/test_metadrive.py --checkpoint checkpoints/best_planner.pt --render

    # Headless (metrics only)
    python scripts/test_metadrive.py --checkpoint checkpoints/best_planner.pt

    # A/B: legacy pure-pursuit vs HMAP Controller on identical seeds
    python scripts/test_metadrive.py --checkpoint checkpoints/best_planner.pt --compare

    # Tune longitudinal gain
    python scripts/test_metadrive.py --checkpoint checkpoints/best_planner.pt --kp 0.2
"""

import argparse
import os
import sys
import math
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halo_planner.model import (
    HaloPlanner, MAX_OBJECTS, MAX_LANES, MAX_LANE_POINTS,
    TRAJECTORY_STEPS, WAYPOINT_DIM, NUM_META_ACTIONS, META_ACTIONS,
)

# HMAP Controller — kinematic trajectory controller (drop next to this script,
# or anywhere on PYTHONPATH).
from hmap_controller import HMAPController


# ---------------------------------------------------------------------------
# Coordinate transform (world → ego frame)
# ---------------------------------------------------------------------------
def world_to_ego(world_pos, ego_pos, ego_heading):
    """Transform world position(s) to ego vehicle frame."""
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    if world_pos.ndim == 1:
        return R @ (world_pos - ego_pos)
    else:
        return (R @ (world_pos - ego_pos).T).T


# ---------------------------------------------------------------------------
# MetaDrive → Planner input adapter
# ---------------------------------------------------------------------------
def extract_ego_state(agent):
    """Extract ego state matching EgomotionData format (9 floats)."""
    ego_heading = agent.heading_theta
    world_vel = np.array(agent.velocity, dtype=np.float64)

    # Rotate world velocity to body frame
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    body_vel_x = cos_h * world_vel[0] - sin_h * world_vel[1]
    body_vel_y = sin_h * world_vel[0] + cos_h * world_vel[1]

    return np.array([
        body_vel_x, body_vel_y, 0.0,   # velX, velY, velZ (body frame m/s)
        0.0, 0.0, 0.0,                 # angVelX, angVelY, angVelZ
        0.0, 0.0, 0.0,                 # position (ego = origin)
    ], dtype=np.float32)


def extract_objects(env, max_dist=80.0):
    """
    Extract traffic vehicles as FusedDetection-compatible tensors.
    Returns: objects (MAX_OBJECTS, 16), mask (MAX_OBJECTS,)
    """
    agent = env.agent
    ego_pos = np.array(agent.position, dtype=np.float64)
    ego_heading = agent.heading_theta

    objects = np.zeros((MAX_OBJECTS, 16), dtype=np.float32)
    mask = np.zeros(MAX_OBJECTS, dtype=bool)

    tm = env.engine.traffic_manager
    count = 0

    for vid, vehicle in tm.spawned_objects.items():
        if count >= MAX_OBJECTS:
            break

        world_pos = np.array(vehicle.position, dtype=np.float64)
        local_pos = world_to_ego(world_pos, ego_pos, ego_heading)

        dist = np.linalg.norm(local_pos)
        if dist > max_dist or dist < 1.0:
            continue

        # Vehicle dimensions
        w = getattr(vehicle, "WIDTH", 1.85)
        l = getattr(vehicle, "LENGTH", 4.5)
        h = getattr(vehicle, "DEFAULT_HEIGHT", 1.19)

        # Relative heading
        rel_heading = vehicle.heading_theta - ego_heading

        # Velocity in ego frame
        world_vel = np.array(vehicle.velocity, dtype=np.float64)
        local_vel = world_to_ego(world_vel + ego_pos, ego_pos, ego_heading)

        # Pack into 16-float FusedDetection vector
        objects[count] = [
            local_pos[0], local_pos[1], 0.0,   # x, y, z
            w, l, h,                            # dimensions
            rel_heading,                        # rotation
            1.0, 0.0, 0.0,                     # class one-hot: vehicle
            1.0,                                # fusedConfidence
            1.0,                                # isDynamic
            1.0,                                # hasLidarSource
            1.0,                                # hasCameraSource
            local_vel[0], local_vel[1],         # velX, velY
        ]
        mask[count] = True
        count += 1

    return objects, mask


def extract_lanes(env, max_lanes=MAX_LANES):
    """
    Extract lane centerlines from road network.
    Returns: lanes (MAX_LANES, 43), mask (MAX_LANES,)
    """
    agent = env.agent
    ego_pos = np.array(agent.position, dtype=np.float64)
    ego_heading = agent.heading_theta

    lanes_out = np.zeros((MAX_LANES, MAX_LANE_POINTS * 2 + 3), dtype=np.float32)
    mask = np.zeros(MAX_LANES, dtype=bool)

    # Collect lanes from current + next road segments
    all_lanes = []

    # Reference lanes (current road segment)
    if hasattr(agent, "reference_lanes") and agent.reference_lanes:
        all_lanes.extend(agent.reference_lanes)

    # Next road segment lanes
    navi = agent.navigation
    if hasattr(navi, "next_ref_lanes") and navi.next_ref_lanes:
        all_lanes.extend(navi.next_ref_lanes)

    # Also try to get lanes from nearby roads in the graph
    try:
        road_network = env.current_map.road_network
        for road_from, connections in road_network.graph.items():
            for road_to, lane_list in connections.items():
                for lane in lane_list:
                    if lane not in all_lanes:
                        # Check if lane is near ego
                        mid_pos = np.array(lane.position(lane.length / 2, 0), dtype=np.float64)
                        if np.linalg.norm(mid_pos - ego_pos) < 60.0:
                            all_lanes.append(lane)
    except Exception:
        pass

    # Deduplicate and limit
    seen = set()
    unique_lanes = []
    for lane in all_lanes:
        lid = id(lane)
        if lid not in seen:
            seen.add(lid)
            unique_lanes.append(lane)
    all_lanes = unique_lanes[:max_lanes]

    for i, lane in enumerate(all_lanes):
        try:
            # Sample centerline points
            num_samples = min(MAX_LANE_POINTS, max(2, int(lane.length / 2.0)))
            s_values = np.linspace(0, lane.length, MAX_LANE_POINTS)

            points_world = np.array([lane.position(s, 0) for s in s_values], dtype=np.float64)
            points_ego = world_to_ego(points_world, ego_pos, ego_heading)

            # Flatten to (MAX_LANE_POINTS * 2,)
            flat_pts = points_ego.flatten().astype(np.float32)

            # Lane type one-hot [driving, connector, other]
            lane_type = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            lanes_out[i, :MAX_LANE_POINTS * 2] = flat_pts
            lanes_out[i, MAX_LANE_POINTS * 2:] = lane_type
            mask[i] = True
        except Exception:
            continue

    return lanes_out, mask


def extract_nav_command(agent):
    """Derive navigation command from next checkpoint direction."""
    navi = agent.navigation

    # Use navi_arrow_dir if available
    if hasattr(navi, "navi_arrow_dir"):
        arrow = navi.navi_arrow_dir
        if arrow is not None and len(arrow) == 2:
            # Transform arrow direction to ego frame
            ego_heading = agent.heading_theta
            cos_h = np.cos(-ego_heading)
            sin_h = np.sin(-ego_heading)
            local_y = sin_h * arrow[0] + cos_h * arrow[1]

            if local_y > 0.3:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # left
            elif local_y < -0.3:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # right

    return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # straight


def scene_to_tensors(env, device):
    """Convert full MetaDrive scene to planner input tensors."""
    agent = env.agent

    ego_state = extract_ego_state(agent)
    objects, obj_mask = extract_objects(env)
    lanes, lane_mask = extract_lanes(env)
    nav_command = extract_nav_command(agent)

    # To torch tensors, add batch dimension
    return {
        "ego_state": torch.from_numpy(ego_state).unsqueeze(0).to(device),
        "objects": torch.from_numpy(objects).unsqueeze(0).to(device),
        "object_mask": torch.from_numpy(obj_mask).unsqueeze(0).to(device),
        "lanes": torch.from_numpy(lanes).unsqueeze(0).to(device),
        "lane_mask": torch.from_numpy(lane_mask).unsqueeze(0).to(device),
        "nav_command": torch.from_numpy(nav_command).unsqueeze(0).to(device),
    }


# ---------------------------------------------------------------------------
# LEGACY controller — retained ONLY as the --compare baseline.
# The HMAP Controller (hmap_controller.py) is the real controller.
# ---------------------------------------------------------------------------
def waypoints_to_action(waypoints, agent, lookahead_index=8):
    # Speed-adaptive lookahead
    current_speed = np.linalg.norm(agent.velocity)
    adaptive_idx = int(np.clip(5 + current_speed * 1.2, 5, 20))
    idx = min(adaptive_idx, len(waypoints) - 1)
    target_x = waypoints[idx, 0]
    target_y = waypoints[idx, 1]

    # Pure pursuit steering
    ld = max(np.sqrt(target_x**2 + target_y**2), 0.1)
    alpha = np.arctan2(target_y, target_x)
    wheelbase = agent.FRONT_WHEELBASE + agent.REAR_WHEELBASE
    steering_angle = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)
    max_steer_rad = np.radians(agent.max_steering)
    steering = np.clip(steering_angle / max_steer_rad, -1.0, 1.0)

    # Lane centering — stronger correction
    lane_correction = -0.5 * waypoints[idx, 1] / max(ld, 1.0)
    steering = np.clip(steering + lane_correction, -1.0, 1.0)

    # Target speed from trajectory extent — use WP[39] (full 4s horizon)
    far_idx = min(39, len(waypoints) - 1)
    far_dist = np.sqrt(waypoints[far_idx, 0]**2 + waypoints[far_idx, 1]**2)
    far_time = (far_idx + 1) * 0.1
    target_speed = far_dist / far_time

    speed_error = target_speed - current_speed

    if current_speed < 1.0 and waypoints[5, 0] > 1.0:
        throttle = 0.8
    elif speed_error > 0:
        throttle = np.clip(0.3 + speed_error / 5.0, 0.2, 1.0)
    else:
        throttle = np.clip(speed_error / 8.0, -0.3, 0.0)

    return [float(steering), float(throttle)]


# ---------------------------------------------------------------------------
# Debug-frame dump — print planner INPUTS (what it saw) and OUTPUTS (what it
# produced) side by side, to localise plumbing bug vs closed-loop gap.
# ---------------------------------------------------------------------------
def _dump_debug_frame(env, inputs, waypoints, meta_logits, step, plan_count):
    agent = env.agent
    print(f"\n  ---- DEBUG FRAME (plan #{plan_count}, sim step {step}) ----")

    # === GROUND TRUTH from MetaDrive ===
    pos = np.array(agent.position)
    print(f"  [truth] ego world pos={pos.round(2)} heading={agent.heading_theta:+.3f} "
          f"speed={np.linalg.norm(agent.velocity):.2f} m/s")
    # Lateral offset from lane centre + lane heading — the on-road reference.
    try:
        lane = agent.lane
        long, lat = lane.local_coordinates(agent.position)
        lane_heading = lane.heading_theta_at(long)
        print(f"  [truth] lane lateral offset={lat:+.2f}m (0=centre)  "
              f"lane heading={lane_heading:+.3f}  rel={_wrap(lane_heading-agent.heading_theta):+.3f}")
    except Exception as e:
        print(f"  [truth] lane query failed: {e}")

    # === INPUTS the planner received ===
    ego_in = inputs["ego_state"][0].cpu().numpy()
    objs = inputs["objects"][0].cpu().numpy()
    obj_m = inputs["object_mask"][0].cpu().numpy()
    lanes = inputs["lanes"][0].cpu().numpy()
    lane_m = inputs["lane_mask"][0].cpu().numpy()
    nav = inputs["nav_command"][0].cpu().numpy()
    print(f"  [in] ego_state={ego_in.round(2)}")
    print(f"  [in] n_objects={int(obj_m.sum())}  n_lanes={int(lane_m.sum())}  "
          f"nav={nav.round(1)} (L,S,R)")
    # Lane y-signs: on a straight road the nearest lane centre should bracket
    # the ego symmetrically; a consistent y-sign reveals a frame flip on input.
    if lane_m.sum() > 0:
        first_lane = lanes[np.argmax(lane_m)]
        pts = first_lane[:40].reshape(-1, 2)  # MAX_LANE_POINTS x 2
        print(f"  [in] lane0 ego-frame: x {pts[:,0].min():+.1f}..{pts[:,0].max():+.1f}  "
              f"y {pts[:,1].min():+.1f}..{pts[:,1].max():+.1f}  "
              f"(y~0 = lane runs ahead; y all one sign = offset/flip)")

    # === OUTPUTS the planner produced ===
    wp = waypoints
    probs = np.exp(meta_logits) / np.exp(meta_logits).sum()
    print(f"  [out] WP0=({wp[0,0]:+.1f},{wp[0,1]:+.1f}) "
          f"WP20=({wp[20,0]:+.1f},{wp[20,1]:+.1f}) "
          f"WP39=({wp[39,0]:+.1f},{wp[39,1]:+.1f})")
    print(f"  [out] x {wp[:,0].min():+.1f}..{wp[:,0].max():+.1f}  "
          f"y {wp[:,1].min():+.1f}..{wp[:,1].max():+.1f}  "
          f"theta {wp[:,2].min():+.2f}..{wp[:,2].max():+.2f}  "
          f"vel {wp[:,3].min():+.1f}..{wp[:,3].max():+.1f}")
    meta_names = ["follow", "lc_left", "lc_right", "stop", "yield", "reverse"]
    top = int(np.argmax(probs))
    print(f"  [out] meta={meta_names[top]} ({probs[top]:.2f})  all={probs.round(2)}")
    # Interpretation hint
    fwd = wp[39, 0]
    lat = wp[39, 1]
    print(f"  [read] forward extent={fwd:+.1f}m over 4s -> implied {fwd/4:.1f} m/s; "
          f"lateral drift at horizon={lat:+.1f}m "
          f"({'RIGHT' if lat < -0.5 else 'LEFT' if lat > 0.5 else 'straight'})")
    print(f"  -------------------------------------------------------")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Single-episode runner — used by both normal and --compare modes.
# `controller_kind` is "hmap" or "legacy". Returns a result dict.
# ---------------------------------------------------------------------------
def run_episode(env, model, device, args, controller_kind, dt, verbose=True):
    obs, info = env.reset()

    # --- Moving start (in-distribution hand-off) ---------------------------
    # nuScenes training data essentially never contains a standstill on an empty
    # road, so the planner cannot launch from 0 m/s (verified: y-drift -1.3m at
    # 5 m/s vs -0.04m at 16 m/s). We drive open-loop with fixed throttle until
    # the ego reaches ~warmup_speed, THEN hand control to the planner — so it
    # operates in the speed regime it was trained on. This is a test-harness
    # accommodation for a known training-data gap, not a model fix.
    if args.warmup_speed > 0.0:
        for _ in range(args.warmup_max_steps):
            if np.linalg.norm(env.agent.velocity) >= args.warmup_speed:
                break
            obs, _, term, trunc, info = env.step([0.0, 1.0])  # straight, full throttle
            if term or trunc:
                obs, info = env.reset()
                break
        if verbose:
            print(f"  [warmup] handing off to planner at "
                  f"{np.linalg.norm(env.agent.velocity):.1f} m/s")

    # Build the controller for this episode.
    hmap = None
    if controller_kind == "hmap":
        hmap = HMAPController(env.agent, kp=args.kp,
                              enable_rollout_check=args.rollout_check)

    total_distance = 0.0
    total_reward = 0.0
    collisions = 0
    steps = 0
    out_of_road = False
    reached_dest = False
    current_waypoints = None
    current_meta_logits = None
    plan_count = 0
    inference_times = []
    # Sanity-verdict counters (HMAP only).
    verdict_counts = {"ok": 0, "clamp": 0, "reject": 0}
    clamp_reasons = {}

    last_pos = np.array(env.agent.position, dtype=np.float64)

    for step in range(args.max_steps):
        # --- Re-plan at 10Hz (every replan_every sim steps) ---
        if step % args.replan_every == 0:
            with torch.no_grad():
                t0 = time.time()
                inputs = scene_to_tensors(env, device)
                out = model(**inputs)
                dt_inf = time.time() - t0
                inference_times.append(dt_inf * 1000)

            current_waypoints = out["waypoints"][0].cpu().numpy()  # (40, 4)
            current_meta_logits = out["meta_logits"][0].cpu().numpy()  # (6,)
            plan_count += 1

            # --- Debug-frame dump: inputs the planner SAW + outputs it PRODUCED.
            # Settles plumbing-bug vs closed-loop-gap. Dumps the first few plans.
            if args.debug_frame and plan_count <= args.debug_frame:
                _dump_debug_frame(env, inputs, current_waypoints,
                                  current_meta_logits, step, plan_count)

        # --- Convert waypoints to steering/throttle ---
        if current_waypoints is not None:
            if controller_kind == "hmap":
                action = hmap.compute_action(
                    current_waypoints, current_meta_logits, env.agent, dt=dt
                )
                v, reason, streak = hmap.last_verdict
                verdict_counts[v] = verdict_counts.get(v, 0) + 1
                if v == "clamp" and reason:
                    clamp_reasons[reason] = clamp_reasons.get(reason, 0) + 1
                if verbose and v != "ok" and step % args.replan_every == 0:
                    print(f"  [sanity] step {step:4d}: {v} ({reason}) streak={streak}")
            else:
                action = waypoints_to_action(current_waypoints, env.agent)
        else:
            action = [0.0, 0.0]

        # --- Step simulator ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Track metrics ---
        current_pos = np.array(env.agent.position, dtype=np.float64)
        total_distance += np.linalg.norm(current_pos - last_pos)
        total_reward += reward
        last_pos = current_pos
        steps += 1

        if info.get("crash_vehicle", False) or info.get("crash_object", False):
            collisions += 1

        if args.topdown:
            env.render(mode="topdown", window=True, screen_size=(600, 600),
                       screen_record=args.save_video is not None)

        if terminated or truncated:
            reached_dest = info.get("arrive_dest", False)
            out_of_road = info.get("out_of_road", False)
            break

    route_completion = env.agent.navigation.route_completion
    avg_inference = np.mean(inference_times) if inference_times else 0

    return {
        "controller": controller_kind,
        "steps": steps,
        "distance_m": total_distance,
        "reward": total_reward,
        "collisions": collisions,
        "route_completion": route_completion,
        "reached_dest": reached_dest,
        "out_of_road": out_of_road,
        "avg_inference_ms": avg_inference,
        "num_replans": plan_count,
        "verdict_counts": verdict_counts,
        "clamp_reasons": clamp_reasons,
    }


def print_episode(result, ep, n):
    status = ("ARRIVED" if result["reached_dest"]
              else "OUT OF ROAD" if result["out_of_road"]
              else "CRASHED" if result["collisions"] > 0
              else "TIMEOUT")
    print(f"\n  [{result['controller']}] Episode {ep+1}/{n}: {status}")
    print(f"    Distance:         {result['distance_m']:.1f}m")
    print(f"    Route completion: {result['route_completion'] * 100:.1f}%")
    print(f"    Collisions:       {result['collisions']}")
    print(f"    Avg inference:    {result['avg_inference_ms']:.1f}ms")
    if result["controller"] == "hmap":
        vc = result["verdict_counts"]
        print(f"    Sanity:           ok={vc.get('ok',0)} "
              f"clamp={vc.get('clamp',0)} reject={vc.get('reject',0)}")
        if result["clamp_reasons"]:
            print(f"    Clamp reasons:    {result['clamp_reasons']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HALO Planner — MetaDrive Test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_planner.pt")
    parser.add_argument("--render", action="store_true", help="3D rendered window")
    parser.add_argument("--topdown", action="store_true", help="Top-down pygame view")
    parser.add_argument("--num_scenarios", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--replan_every", type=int, default=4,
                        help="Re-run planner every N sim steps (sim=40Hz, planner=10Hz)")
    parser.add_argument("--map", type=str, default="SCCS",
                        help="Map layout: S=straight, C=curve, X=intersection, O=roundabout, T=T-junction")
    parser.add_argument("--traffic_density", type=float, default=0.3)
    parser.add_argument("--save_video", type=str, default=None, help="Path to save top-down video")
    # --- HMAP Controller options ---
    parser.add_argument("--kp", type=float, default=0.2,
                        help="Longitudinal PID proportional gain (tune this first)")
    parser.add_argument("--rollout_check", action="store_true",
                        help="Enable advisory forward-rollout divergence check (Concern B)")
    parser.add_argument("--debug-frame", type=int, default=0, dest="debug_frame",
                        help="Dump planner inputs+outputs for the first N plans (0=off)")
    parser.add_argument("--warmup-speed", type=float, default=12.0, dest="warmup_speed",
                        help="Drive open-loop until ego reaches this speed (m/s) before "
                             "handing to the planner (0=disable; nuScenes-trained planner "
                             "cannot launch from standstill)")
    parser.add_argument("--warmup-max-steps", type=int, default=60, dest="warmup_max_steps",
                        help="Max sim steps to spend reaching warmup speed")
    parser.add_argument("--compare", action="store_true",
                        help="Run legacy pure-pursuit AND HMAP Controller on identical seeds")
    parser.add_argument("--controller", type=str, default="hmap",
                        choices=["hmap", "legacy"],
                        help="Which controller to use when not in --compare mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    print(f"Loading model from {args.checkpoint}...")
    model = HaloPlanner()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Model loaded (epoch {ckpt['epoch']}, val ADE: {ckpt.get('val_ade', '?')}m)")
    print(f"  Parameters: {model.count_parameters() / 1e6:.1f}M")

    # --- Sim tick period for the PID. sim≈40Hz with replan_every=4 -> 10Hz plan.
    # The control action is applied every sim step, so dt = 1/40 = 0.025s.
    dt = 0.025

    # --- Create environment ---
    from metadrive.envs.metadrive_env import MetaDriveEnv

    env_config = dict(
        map=args.map,
        use_render=args.render,
        num_scenarios=args.num_scenarios,
        traffic_density=args.traffic_density,
        start_seed=42,   # fixed seed base -> identical scenarios across controllers
    )

    print(f"\nCreating MetaDrive environment...")
    print(f"  Map: {args.map} | Traffic: {args.traffic_density} | "
          f"Render: {'3D' if args.render else 'topdown' if args.topdown else 'headless'}")
    if args.compare:
        print(f"  Mode: COMPARE (legacy vs hmap), kp={args.kp}")
    else:
        print(f"  Mode: {args.controller.upper()}, kp={args.kp}")
    env = MetaDriveEnv(env_config)

    # --- Run episodes ---
    # In compare mode we run each scenario twice (same seed) — once per
    # controller — so the only variable is the controller.
    kinds = ["legacy", "hmap"] if args.compare else [args.controller]
    results = {k: [] for k in kinds}

    for ep in range(args.num_scenarios):
        print(f"\n{'='*52}\nScenario {ep + 1}/{args.num_scenarios}\n{'='*52}")
        for kind in kinds:
            # reset() with the same episode index reproduces the same scenario
            # because start_seed is fixed and num_scenarios bounds the seed set.
            res = run_episode(env, model, device, args, kind, dt, verbose=not args.compare)
            results[kind].append(res)
            print_episode(res, ep, args.num_scenarios)

    if args.topdown and args.save_video:
        try:
            env.top_down_renderer.generate_gif(args.save_video)
            print(f"\nSaved video to {args.save_video}")
        except Exception as e:
            print(f"\nCould not save video: {e}")

    env.close()

    # --- Summary ---
    print(f"\n{'='*52}\nSUMMARY ({args.num_scenarios} scenarios)\n{'='*52}")

    def summarise(rs):
        return {
            "dist": np.mean([r["distance_m"] for r in rs]),
            "route": np.mean([r["route_completion"] for r in rs]) * 100,
            "coll": sum(r["collisions"] for r in rs),
            "arr": sum(1 for r in rs if r["reached_dest"]),
            "infer": np.mean([r["avg_inference_ms"] for r in rs]),
        }

    if args.compare:
        L, H = summarise(results["legacy"]), summarise(results["hmap"])
        # Aggregate HMAP sanity verdicts across episodes.
        agg = {"ok": 0, "clamp": 0, "reject": 0}
        for r in results["hmap"]:
            for k, v in r["verdict_counts"].items():
                agg[k] = agg.get(k, 0) + v
        print(f"  {'metric':<20}{'legacy':>12}{'hmap':>12}")
        print(f"  {'-'*44}")
        print(f"  {'avg distance (m)':<20}{L['dist']:>12.1f}{H['dist']:>12.1f}")
        print(f"  {'avg route (%)':<20}{L['route']:>12.1f}{H['route']:>12.1f}")
        print(f"  {'total collisions':<20}{L['coll']:>12d}{H['coll']:>12d}")
        print(f"  {'arrivals':<20}{L['arr']:>12d}{H['arr']:>12d}")
        print(f"  {'avg inference (ms)':<20}{L['infer']:>12.1f}{H['infer']:>12.1f}")
        print(f"\n  HMAP sanity totals: {agg}")
    else:
        S = summarise(results[args.controller])
        print(f"  Controller:       {args.controller}")
        print(f"  Avg distance:     {S['dist']:.1f}m")
        print(f"  Avg route:        {S['route']:.1f}%")
        print(f"  Total collisions: {S['coll']}")
        print(f"  Arrivals:         {S['arr']}/{args.num_scenarios}")
        print(f"  Avg inference:    {S['infer']:.1f}ms ({1000/max(S['infer'],1):.0f} Hz)")


if __name__ == "__main__":
    main()