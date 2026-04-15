"""
HALO Planner — MetaDrive closed-loop evaluation.

Runs the trained planner in MetaDrive simulator:
  1. Reads ego/traffic/lane state from MetaDrive (replaces FusionEngine)
  2. Converts to planner input tensors (same format as FusedPacket)
  3. Runs inference through best_planner.pt
  4. Converts output waypoints to steering/throttle via pure pursuit
  5. Steps the simulator — full closed loop

Usage:
    # 3D rendered window (needs display)
    python scripts/test_metadrive.py --checkpoint checkpoints/best_planner.pt --render

    # Top-down view saved to video
    python scripts/test_metadrive.py --checkpoint checkpoints/best_planner.pt --topdown

    # Headless (metrics only)
    python scripts/test_metadrive.py --checkpoint checkpoints/best_planner.pt
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
# Trajectory → MetaDrive action (pure pursuit controller)
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
# Main closed-loop evaluation
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

    # --- Create environment ---
    from metadrive.envs.metadrive_env import MetaDriveEnv

    env_config = dict(
        map=args.map,
        use_render=args.render,
        num_scenarios=args.num_scenarios,
        traffic_density=args.traffic_density,
        start_seed=42,
    )

    print(f"\nCreating MetaDrive environment...")
    print(f"  Map: {args.map}")
    print(f"  Traffic density: {args.traffic_density}")
    print(f"  Render: {'3D' if args.render else 'topdown' if args.topdown else 'headless'}")
    env = MetaDriveEnv(env_config)

    # --- Run episodes ---
    all_results = []

    for ep in range(args.num_scenarios):
        obs, info = env.reset()

        total_distance = 0.0
        total_reward = 0.0
        collisions = 0
        steps = 0
        out_of_road = False
        reached_dest = False
        current_waypoints = None
        plan_count = 0
        inference_times = []

        last_pos = np.array(env.agent.position, dtype=np.float64)

        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{args.num_scenarios}")
        print(f"{'='*50}")

        for step in range(args.max_steps):
            # --- Re-plan at 10Hz (every replan_every sim steps) ---
            if step % args.replan_every == 0:
                with torch.no_grad():
                    t0 = time.time()
                    inputs = scene_to_tensors(env, device)
                    out = model(**inputs)
                    dt = time.time() - t0
                    inference_times.append(dt * 1000)
                
                current_waypoints = out["waypoints"][0].cpu().numpy()  # (40, 4)
                meta_idx = out["meta_logits"][0].argmax().item()
                plan_count += 1

                if step % (args.replan_every * 20) == 0:
                    n_obj = inputs["object_mask"][0].sum().item()
                    n_lane = inputs["lane_mask"][0].sum().item()
                    speed = np.linalg.norm(env.agent.velocity)
                    wp = current_waypoints
                    print(f"  Step {step:4d} | Speed: {speed:.1f} m/s | "
                          f"Objects: {n_obj:.0f} | Lanes: {n_lane:.0f} | "
                          f"Action: {META_ACTIONS[meta_idx]} | "
                          f"Inference: {dt*1000:.1f}ms")
                    print(f"    WP[0]:  x={wp[0,0]:+.3f} y={wp[0,1]:+.3f} vel={wp[0,3]:.2f}")
                    print(f"    WP[5]:  x={wp[5,0]:+.3f} y={wp[5,1]:+.3f} vel={wp[5,3]:.2f}")
                    print(f"    WP[20]: x={wp[20,0]:+.3f} y={wp[20,1]:+.3f} vel={wp[20,3]:.2f}")
                    print(f"    WP[39]: x={wp[39,0]:+.3f} y={wp[39,1]:+.3f} vel={wp[39,3]:.2f}")
                    action_preview = waypoints_to_action(current_waypoints, env.agent)
                    print(f"    Action: steer={action_preview[0]:+.3f} throttle={action_preview[1]:+.3f}")

            # --- Convert waypoints to steering/throttle ---
            if current_waypoints is not None:
                action = waypoints_to_action(current_waypoints, env.agent)
            else:
                action = [0.0, 0.0]

            # --- Step simulator ---
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Track metrics ---
            current_pos = np.array(env.agent.position, dtype=np.float64)
            step_dist = np.linalg.norm(current_pos - last_pos)
            total_distance += step_dist
            total_reward += reward
            last_pos = current_pos
            steps += 1

            if info.get("crash_vehicle", False) or info.get("crash_object", False):
                collisions += 1

            # --- Render top-down if requested ---
            if args.topdown:
                env.render(
                    mode="topdown",
                    window=True,
                    screen_size=(600, 600),
                    screen_record=args.save_video is not None,
                )

            if terminated or truncated:
                reached_dest = info.get("arrive_dest", False)
                out_of_road = info.get("out_of_road", False)
                break

        # --- Episode summary ---
        route_completion = env.agent.navigation.route_completion
        avg_inference = np.mean(inference_times) if inference_times else 0

        result = {
            "episode": ep + 1,
            "steps": steps,
            "distance_m": total_distance,
            "reward": total_reward,
            "collisions": collisions,
            "route_completion": route_completion,
            "reached_dest": reached_dest,
            "out_of_road": out_of_road,
            "avg_inference_ms": avg_inference,
            "num_replans": plan_count,
        }
        all_results.append(result)

        status = "ARRIVED" if reached_dest else "OUT OF ROAD" if out_of_road else "TIMEOUT" if steps >= args.max_steps else "CRASHED"
        print(f"\n  Result: {status}")
        print(f"  Distance:         {total_distance:.1f}m")
        print(f"  Route completion: {route_completion * 100:.1f}%")
        print(f"  Collisions:       {collisions}")
        print(f"  Reward:           {total_reward:.2f}")
        print(f"  Avg inference:    {avg_inference:.1f}ms")
        print(f"  Re-plans:         {plan_count}")

    # Save top-down video if requested
    if args.topdown and args.save_video:
        try:
            env.top_down_renderer.generate_gif(args.save_video)
            print(f"\nSaved video to {args.save_video}")
        except Exception as e:
            print(f"\nCould not save video: {e}")

    env.close()

    # --- Final summary ---
    print(f"\n{'='*50}")
    print(f"SUMMARY ({len(all_results)} episodes)")
    print(f"{'='*50}")

    avg_dist = np.mean([r["distance_m"] for r in all_results])
    avg_route = np.mean([r["route_completion"] for r in all_results])
    total_collisions = sum(r["collisions"] for r in all_results)
    arrivals = sum(1 for r in all_results if r["reached_dest"])
    avg_infer = np.mean([r["avg_inference_ms"] for r in all_results])

    print(f"  Avg distance:     {avg_dist:.1f}m")
    print(f"  Avg route:        {avg_route * 100:.1f}%")
    print(f"  Total collisions: {total_collisions}")
    print(f"  Arrivals:         {arrivals}/{len(all_results)}")
    print(f"  Avg inference:    {avg_infer:.1f}ms ({1000/max(avg_infer, 1):.0f} Hz)")


if __name__ == "__main__":
    main()
