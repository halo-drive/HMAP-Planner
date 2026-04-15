"""
HALO Planner — Dataset and model verification.

Run this BEFORE training to catch issues early.
Validates:
    1. nuScenes dataset loads correctly
    2. Map expansion lane queries work
    3. Tensor shapes match model expectations
    4. Value ranges are sane (positions in metres, velocities realistic)
    5. GT trajectory is properly interpolated 2Hz → 10Hz
    6. FusedPacket field mapping is correct
    7. Full forward pass + loss computation works on real data

Usage:
    cd ~/halo_planner
    source planner_env/bin/activate
    python scripts/verify_dataset.py --dataroot data/nuscenes
"""

import argparse
import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halo_planner.model import (
    HaloPlanner, MAX_OBJECTS, MAX_LANES, MAX_LANE_POINTS,
    TRAJECTORY_STEPS, WAYPOINT_DIM, NUM_META_ACTIONS,
)
from halo_planner.losses import PlannerLoss
from halo_planner.dataset_nuscenes import NuScenesDataset


def check(condition: bool, msg: str):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise AssertionError(f"Verification failed: {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to validate")
    args = parser.parse_args()

    print("=" * 60)
    print("HALO Planner — Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Dataset loading
    # ------------------------------------------------------------------
    print("\n[1/7] Loading dataset...")
    t0 = time.time()
    ds = NuScenesDataset(
        dataroot=args.dataroot,
        version=args.version,
        split="train",
    )
    dt = time.time() - t0
    print(f"  Loaded in {dt:.1f}s")
    check(len(ds) > 0, f"Dataset has {len(ds)} samples")
    check(len(ds) > 1000, f"Trainval should have >1000 samples, got {len(ds)}")

    # ------------------------------------------------------------------
    # 2. Single sample extraction
    # ------------------------------------------------------------------
    print("\n[2/7] Extracting sample 0...")
    t0 = time.time()
    sample = ds[0]
    dt = time.time() - t0
    print(f"  Extracted in {dt:.3f}s")

    # ------------------------------------------------------------------
    # 3. Tensor shape verification
    # ------------------------------------------------------------------
    print("\n[3/7] Tensor shapes...")
    check(sample["ego_state"].shape == (9,),
          f"ego_state: {sample['ego_state'].shape} == (9,)")
    check(sample["objects"].shape == (MAX_OBJECTS, 16),
          f"objects: {sample['objects'].shape} == ({MAX_OBJECTS}, 16)")
    check(sample["object_mask"].shape == (MAX_OBJECTS,),
          f"object_mask: {sample['object_mask'].shape} == ({MAX_OBJECTS},)")
    check(sample["lanes"].shape == (MAX_LANES, MAX_LANE_POINTS * 2 + 3),
          f"lanes: {sample['lanes'].shape} == ({MAX_LANES}, {MAX_LANE_POINTS * 2 + 3})")
    check(sample["lane_mask"].shape == (MAX_LANES,),
          f"lane_mask: {sample['lane_mask'].shape} == ({MAX_LANES},)")
    check(sample["nav_command"].shape == (3,),
          f"nav_command: {sample['nav_command'].shape} == (3,)")
    check(sample["gt_waypoints"].shape == (TRAJECTORY_STEPS, WAYPOINT_DIM),
          f"gt_waypoints: {sample['gt_waypoints'].shape} == ({TRAJECTORY_STEPS}, {WAYPOINT_DIM})")
    check(sample["waypoint_mask"].shape == (TRAJECTORY_STEPS,),
          f"waypoint_mask: {sample['waypoint_mask'].shape} == ({TRAJECTORY_STEPS},)")
    check(sample["meta_action"].shape == (),
          f"meta_action: scalar, value={sample['meta_action'].item()}")

    # ------------------------------------------------------------------
    # 4. Value range checks
    # ------------------------------------------------------------------
    print("\n[4/7] Value ranges...")

    # Ego velocity
    ego = sample["ego_state"]
    vel_x, vel_y = ego[0].item(), ego[1].item()
    ego_speed = np.sqrt(vel_x**2 + vel_y**2)
    print(f"  Ego velocity: ({vel_x:.2f}, {vel_y:.2f}) m/s = {ego_speed:.1f} m/s")
    check(ego_speed < 50.0, f"Ego speed {ego_speed:.1f} m/s is plausible (<50)")

    # Object positions
    obj_mask = sample["object_mask"]
    num_objects = obj_mask.sum().item()
    print(f"  Objects detected: {num_objects}")
    check(num_objects > 0, "At least 1 object in sample")
    if num_objects > 0:
        valid_objs = sample["objects"][obj_mask]
        obj_x = valid_objs[:, 0]
        obj_y = valid_objs[:, 1]
        obj_dist = torch.sqrt(obj_x**2 + obj_y**2)
        print(f"  Object distances: min={obj_dist.min():.1f}m, max={obj_dist.max():.1f}m, mean={obj_dist.mean():.1f}m")
        check(obj_dist.max() <= 80.0, f"All objects within 80m cutoff (max={obj_dist.max():.1f}m)")

        # Check class distribution
        class_ids = valid_objs[:, 7:10]  # one-hot columns
        n_veh = (class_ids[:, 0] > 0.5).sum().item()
        n_ped = (class_ids[:, 1] > 0.5).sum().item()
        n_cyc = (class_ids[:, 2] > 0.5).sum().item()
        print(f"  Classes: {n_veh} vehicles, {n_ped} pedestrians, {n_cyc} cyclists")

    # FusedPacket field mapping verification
    print("\n  FusedPacket field mapping (obj feature dim=16):")
    print("    [0:3]   x, y, z          → FusedDetection::x/y/z")
    print("    [3:6]   w, l, h          → FusedDetection::width/length/height")
    print("    [6]     rotation          → FusedDetection::rotation")
    print("    [7:10]  class one-hot     → FusedDetection::classId")
    print("    [10]    fusedConfidence   → FusedDetection::fusedConfidence")
    print("    [11]    isDynamic         → FusedDetection::isDynamic")
    print("    [12]    hasLidarSource    → FusedDetection::hasLidarSource")
    print("    [13]    hasCameraSource   → FusedDetection::hasCameraSource")
    print("    [14:16] velX, velY        → ObjectTracker velocity estimate")
    if num_objects > 0:
        check(valid_objs[:, 10].min() >= 0.0 and valid_objs[:, 10].max() <= 1.0,
              "fusedConfidence in [0, 1]")

    # Lanes
    lane_mask = sample["lane_mask"]
    num_lanes = lane_mask.sum().item()
    print(f"\n  Lanes detected: {num_lanes}")
    check(num_lanes > 0, "At least 1 lane in sample")
    if num_lanes > 0:
        valid_lanes = sample["lanes"][lane_mask]
        lane_pts = valid_lanes[:, :MAX_LANE_POINTS*2].reshape(-1, MAX_LANE_POINTS, 2)
        lane_dist = torch.sqrt(lane_pts[..., 0]**2 + lane_pts[..., 1]**2)
        print(f"  Lane point distances: min={lane_dist.min():.1f}m, max={lane_dist.max():.1f}m")
        check(lane_dist.max() < 200.0, "Lane points within reasonable range")

    # GT trajectory
    gt_wp = sample["gt_waypoints"]
    wp_mask = sample["waypoint_mask"]
    num_valid_wp = wp_mask.sum().item()
    print(f"\n  GT waypoints: {num_valid_wp}/{TRAJECTORY_STEPS} valid")
    check(num_valid_wp >= 8, f"At least 8 valid waypoints (got {num_valid_wp})")
    check(num_valid_wp == TRAJECTORY_STEPS, f"All {TRAJECTORY_STEPS} steps interpolated")

    valid_wp = gt_wp[wp_mask]
    if len(valid_wp) > 0:
        max_disp = torch.sqrt(valid_wp[:, 0]**2 + valid_wp[:, 1]**2).max().item()
        max_vel = valid_wp[:, 3].max().item()
        final_wp = valid_wp[-1]
        print(f"  Max displacement: {max_disp:.2f}m (over {num_valid_wp * 0.1:.1f}s)")
        print(f"  Max velocity: {max_vel:.1f} m/s ({max_vel * 3.6:.0f} km/h)")
        print(f"  Final waypoint: x={final_wp[0]:.2f}, y={final_wp[1]:.2f}, hdg={final_wp[2]:.3f} rad, vel={final_wp[3]:.1f} m/s")
        check(max_disp < 200.0, "4s trajectory < 200m (< 180 km/h)")
        check(max_vel < 60.0, f"Max velocity {max_vel:.1f} m/s plausible (< 216 km/h)")

    # Nav command
    nav = sample["nav_command"]
    nav_labels = ["left", "straight", "right"]
    nav_idx = nav.argmax().item()
    print(f"  Nav command: {nav_labels[nav_idx]} {nav.tolist()}")
    check(nav.sum().item() == 1.0, "Nav is valid one-hot")

    # Meta action
    meta = sample["meta_action"].item()
    meta_labels = ["follow_lane", "lane_change_left", "lane_change_right", "stop", "yield", "reverse"]
    print(f"  Meta-action: {meta_labels[meta]} ({meta})")
    check(0 <= meta < NUM_META_ACTIONS, f"Meta-action in valid range [0, {NUM_META_ACTIONS})")

    # ------------------------------------------------------------------
    # 5. Multi-sample statistics
    # ------------------------------------------------------------------
    print(f"\n[5/7] Sampling {args.num_samples} random samples...")
    rng = np.random.RandomState(42)
    indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    obj_counts = []
    lane_counts = []
    wp_counts = []
    speeds = []
    errors = 0

    for i, idx in enumerate(indices):
        try:
            s = ds[idx]
            obj_counts.append(s["object_mask"].sum().item())
            lane_counts.append(s["lane_mask"].sum().item())
            wp_counts.append(s["waypoint_mask"].sum().item())
            vel = s["ego_state"][:2]
            speeds.append(float(torch.sqrt(vel[0]**2 + vel[1]**2)))
        except Exception as e:
            errors += 1
            print(f"  ERROR at sample {idx}: {e}")

    print(f"  Objects per sample:  min={min(obj_counts)}, max={max(obj_counts)}, mean={np.mean(obj_counts):.1f}")
    print(f"  Lanes per sample:    min={min(lane_counts)}, max={max(lane_counts)}, mean={np.mean(lane_counts):.1f}")
    print(f"  Valid WPs per sample: min={min(wp_counts)}, max={max(wp_counts)}, mean={np.mean(wp_counts):.1f}")
    print(f"  Ego speed (m/s):     min={min(speeds):.1f}, max={max(speeds):.1f}, mean={np.mean(speeds):.1f}")
    check(errors == 0, f"All {args.num_samples} samples loaded without errors")

    # ------------------------------------------------------------------
    # 6. DataLoader batch test
    # ------------------------------------------------------------------
    print("\n[6/7] DataLoader batch test...")
    from torch.utils.data import DataLoader, Subset

    subset = Subset(ds, indices[:4])
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    check(batch["ego_state"].shape == (2, 9), f"Batched ego: {batch['ego_state'].shape}")
    check(batch["objects"].shape == (2, MAX_OBJECTS, 16), f"Batched objects: {batch['objects'].shape}")
    check(batch["gt_waypoints"].shape == (2, TRAJECTORY_STEPS, WAYPOINT_DIM),
          f"Batched GT: {batch['gt_waypoints'].shape}")
    print("  Batch collation works correctly")

    # ------------------------------------------------------------------
    # 7. Forward pass + loss on real data
    # ------------------------------------------------------------------
    print("\n[7/7] Model forward pass + loss on real data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = HaloPlanner().to(device)
    criterion = PlannerLoss()
    print(f"  Model: {model.count_parameters()/1e6:.1f}M parameters")

    # Move batch to device
    ego = batch["ego_state"].to(device)
    objs = batch["objects"].to(device)
    obj_mask = batch["object_mask"].to(device)
    lanes = batch["lanes"].to(device)
    lane_mask = batch["lane_mask"].to(device)
    nav = batch["nav_command"].to(device)
    gt_wp = batch["gt_waypoints"].to(device)
    wp_mask = batch["waypoint_mask"].to(device)
    meta = batch["meta_action"].to(device)

    t0 = time.time()
    out = model(ego, objs, obj_mask, lanes, lane_mask, nav)
    dt = time.time() - t0

    check(out["waypoints"].shape == (2, TRAJECTORY_STEPS, WAYPOINT_DIM),
          f"Output waypoints: {out['waypoints'].shape}")
    check(out["meta_logits"].shape == (2, NUM_META_ACTIONS),
          f"Output meta logits: {out['meta_logits'].shape}")
    check(not torch.isnan(out["waypoints"]).any(), "No NaN in waypoints")
    check(not torch.isnan(out["meta_logits"]).any(), "No NaN in meta logits")
    print(f"  Forward pass: {dt*1000:.1f}ms")

    losses = criterion(
        out["waypoints"], gt_wp, out["meta_logits"], meta, wp_mask,
        objs[..., :2], obj_mask,
    )
    print(f"  Total loss: {losses['total'].item():.4f}")
    for k, v in losses.items():
        if k != "total":
            print(f"    {k}: {v.item():.4f}")
    check(not torch.isnan(losses["total"]), "Loss is not NaN")
    check(losses["total"].item() > 0, "Loss is positive")
    check(losses["total"].item() < 100, "Loss is not exploding")

    # Backward pass
    losses["total"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient norm: {grad_norm:.2f}")
    check(grad_norm > 0, "Gradients are flowing")
    check(grad_norm < 1e6, "Gradients are not exploding")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — ready to train")
    print("=" * 60)
    print(f"\nTo start training:")
    print(f"  python scripts/train.py --dataroot {args.dataroot} --version {args.version} --epochs 50 --batch_size 16")


if __name__ == "__main__":
    main()
