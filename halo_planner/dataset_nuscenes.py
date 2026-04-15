"""
HALO Planner — nuScenes dataset adapter.

Converts nuScenes annotations into the structured scene format
matching FusedPacket.hpp for training the planner.

nuScenes provides:
    - 3D bounding boxes (objects)
    - HD map with lane polylines
    - Ego vehicle trajectory (ground truth for imitation learning)
    - Navigation command (derived from trajectory direction)

This adapter extracts all of these into tensor form for the model.
"""

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import (
    MAX_OBJECTS, MAX_LANES, MAX_LANE_POINTS,
    NUM_CLASSES, TRAJECTORY_STEPS, WAYPOINT_DIM,
    NUM_META_ACTIONS,
)


# nuScenes class name → classId (matching FusedPacket convention)
NUSCENES_CLASS_MAP = {
    "car": 0, "truck": 0, "bus": 0, "trailer": 0,
    "construction_vehicle": 0,
    "pedestrian": 1, "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1, "human.pedestrian.construction_worker": 1,
    "human.pedestrian.police_officer": 1,
    "bicycle": 2, "motorcycle": 2,
}


def quaternion_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from quaternion [w, x, y, z]."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_nav_command(future_traj: np.ndarray) -> np.ndarray:
    """
    Derive navigation command from future trajectory.
    Returns one-hot [left, straight, right] based on net lateral displacement.
    """
    if len(future_traj) < 2:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Net lateral displacement (y in ego frame)
    lateral = future_traj[-1, 1] - future_traj[0, 1]

    if lateral > 2.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif lateral < -2.0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def compute_meta_action(future_traj: np.ndarray) -> int:
    """
    Derive meta-action label from future trajectory.
    Simple heuristic — can be refined with map-based logic later.
    """
    if len(future_traj) < 2:
        return 0  # follow_lane

    speed = np.linalg.norm(future_traj[-1, :2] - future_traj[0, :2]) / (len(future_traj) * 0.5)
    lateral = abs(future_traj[-1, 1] - future_traj[0, 1])

    if speed < 0.3:
        return 3  # stop
    elif lateral > 2.0:
        if future_traj[-1, 1] > future_traj[0, 1]:
            return 1  # lane_change_left
        else:
            return 2  # lane_change_right
    else:
        return 0  # follow_lane


class NuScenesDataset(Dataset):
    """
    nuScenes dataset adapter for HALO Planner.

    Requires nuscenes-devkit: pip install nuscenes-devkit

    Args:
        dataroot:   Path to nuScenes dataset root (e.g. /data/nuscenes)
        version:    "v1.0-mini" for prototyping, "v1.0-trainval" for full
        split:      "train" or "val"
        future_secs: Trajectory prediction horizon in seconds (default 4.0)
        past_secs:   Past ego history for context (default 2.0)
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "train",
        future_secs: float = 4.0,
        past_secs: float = 2.0,
    ):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.utils.splits import create_splits_scenes

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.future_secs = future_secs
        self.past_secs = past_secs
        self.annotation_rate = 2.0       # nuScenes keyframe rate (Hz)
        self.output_rate = 10.0          # Planner output rate (Hz)
        self.future_steps_ann = int(future_secs * self.annotation_rate)  # 8 keyframes

        # Get scene names for this split
        splits = create_splits_scenes()
        split_scenes = splits[split] if split in splits else splits["mini_train"]

        # Collect valid sample tokens — trim per-scene to ensure future GT exists
        self.samples = []
        for scene in self.nusc.scene:
            if scene["name"] not in split_scenes:
                continue
            scene_samples = []
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                scene_samples.append(sample_token)
                sample_token = sample["next"] if sample["next"] else None
            # Keep only samples that have enough future within this scene
            if len(scene_samples) > self.future_steps_ann:
                self.samples.extend(scene_samples[:-self.future_steps_ann])

        # Load maps per location
        self.map_cache = {}
        for log in self.nusc.log:
            loc = log["location"]
            if loc not in self.map_cache:
                self.map_cache[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)

        print(f"[NuScenesDataset] Loaded {len(self.samples)} samples ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_ego_pose(self, sample_token: str) -> dict:
        """Get ego pose for a sample."""
        sample = self.nusc.get("sample", sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        return ego

    def _get_future_trajectory(self, sample_token: str) -> np.ndarray:
        """
        Get future ego trajectory as waypoints in the current ego frame.
        Collects at 2Hz (nuScenes keyframe rate), then interpolates to 10Hz.
        Returns (TRAJECTORY_STEPS, 4) — [x, y, heading, velocity].
        """
        current_ego = self._get_ego_pose(sample_token)
        current_pos = np.array(current_ego["translation"][:2])
        current_yaw = quaternion_to_yaw(current_ego["rotation"])

        # Rotation matrix: world → ego frame
        cos_y, sin_y = np.cos(-current_yaw), np.sin(-current_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        # Collect keyframes at 2Hz
        keyframes = []  # [(time_offset, x, y, heading)]
        token = sample_token
        dt_ann = 1.0 / self.annotation_rate  # 0.5s between keyframes
        for i in range(self.future_steps_ann):
            sample = self.nusc.get("sample", token)
            if not sample["next"]:
                break
            token = sample["next"]
            fut_ego = self._get_ego_pose(token)
            fut_pos = np.array(fut_ego["translation"][:2])
            fut_yaw = quaternion_to_yaw(fut_ego["rotation"])

            local_pos = R @ (fut_pos - current_pos)
            local_heading = fut_yaw - current_yaw
            t = (i + 1) * dt_ann

            keyframes.append([t, local_pos[0], local_pos[1], local_heading])

        if len(keyframes) < 2:
            return np.zeros((1, 4), dtype=np.float32)

        keyframes = np.array(keyframes, dtype=np.float32)  # (K, 4): t, x, y, hdg

        # Interpolate to 10Hz
        dt_out = 1.0 / self.output_rate  # 0.1s
        t_out = np.arange(1, TRAJECTORY_STEPS + 1) * dt_out  # 0.1, 0.2, ..., 4.0
        t_out = np.clip(t_out, 0, keyframes[-1, 0])  # don't extrapolate

        x_interp = np.interp(t_out, keyframes[:, 0], keyframes[:, 1])
        y_interp = np.interp(t_out, keyframes[:, 0], keyframes[:, 2])
        hdg_interp = np.interp(t_out, keyframes[:, 0], keyframes[:, 3])

        # Compute velocity from position differences
        vel = np.zeros(TRAJECTORY_STEPS, dtype=np.float32)
        vel[0] = np.sqrt(x_interp[0]**2 + y_interp[0]**2) / dt_out
        for i in range(1, TRAJECTORY_STEPS):
            dx = x_interp[i] - x_interp[i-1]
            dy = y_interp[i] - y_interp[i-1]
            vel[i] = np.sqrt(dx**2 + dy**2) / dt_out

        waypoints = np.stack([x_interp, y_interp, hdg_interp, vel], axis=-1)
        return waypoints.astype(np.float32)  # (TRAJECTORY_STEPS, 4)

    def _get_objects(self, sample_token: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get 3D object annotations in ego frame.
        Returns:
            objects: (MAX_OBJECTS, 16)
            mask: (MAX_OBJECTS,) — True for valid
        """
        sample = self.nusc.get("sample", sample_token)
        current_ego = self._get_ego_pose(sample_token)
        current_pos = np.array(current_ego["translation"][:2])
        current_yaw = quaternion_to_yaw(current_ego["rotation"])
        cos_y, sin_y = np.cos(-current_yaw), np.sin(-current_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        objects = np.zeros((MAX_OBJECTS, 16), dtype=np.float32)
        mask = np.zeros(MAX_OBJECTS, dtype=bool)

        count = 0
        for ann_token in sample["anns"]:
            if count >= MAX_OBJECTS:
                break
            ann = self.nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]

            # Map to our class IDs
            class_id = None
            for key, cid in NUSCENES_CLASS_MAP.items():
                if key in cat:
                    class_id = cid
                    break
            if class_id is None:
                continue

            # Position in ego frame
            world_pos = np.array(ann["translation"][:2])
            local_pos = R @ (world_pos - current_pos)

            # Skip objects too far away (>80m)
            if np.linalg.norm(local_pos) > 80.0:
                continue

            z = ann["translation"][2] - current_ego["translation"][2]
            yaw = quaternion_to_yaw(ann["rotation"]) - current_yaw
            w, l, h = ann["size"]  # width, length, height

            # Class one-hot
            class_onehot = np.zeros(3, dtype=np.float32)
            class_onehot[class_id] = 1.0

            # Build feature vector: [x,y,z, w,l,h, rot, class(3), conf, isDynamic, hasLidar, hasCam, velX, velY]
            objects[count] = [
                local_pos[0], local_pos[1], z,
                w, l, h,
                yaw,
                class_onehot[0], class_onehot[1], class_onehot[2],
                1.0,    # fusedConfidence (GT = 1.0)
                1.0,    # isDynamic (assume true for annotations)
                1.0,    # hasLidarSource
                1.0,    # hasCameraSource
                0.0, 0.0,  # velocity (not available in annotation directly)
            ]
            mask[count] = True
            count += 1

        return objects, mask

    def _get_lanes(self, sample_token: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get nearby lane polylines from the HD map in ego frame.
        Returns:
            lanes: (MAX_LANES, MAX_LANE_POINTS*2 + 3)
            mask: (MAX_LANES,) — True for valid
        """
        sample = self.nusc.get("sample", sample_token)
        log = self.nusc.get("log", self.nusc.get("scene", sample["scene_token"])["log_token"])
        nusc_map = self.map_cache[log["location"]]

        current_ego = self._get_ego_pose(sample_token)
        ego_x, ego_y = current_ego["translation"][0], current_ego["translation"][1]
        current_yaw = quaternion_to_yaw(current_ego["rotation"])
        cos_y, sin_y = np.cos(-current_yaw), np.sin(-current_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        ego_pos = np.array([ego_x, ego_y])

        # Query nearby lane records
        radius = 50.0  # metres
        lane_records = nusc_map.get_records_in_radius(ego_x, ego_y, radius, ["lane", "lane_connector"])
        lane_tokens = lane_records.get("lane", []) + lane_records.get("lane_connector", [])

        lanes = np.zeros((MAX_LANES, MAX_LANE_POINTS * 2 + 3), dtype=np.float32)
        mask = np.zeros(MAX_LANES, dtype=bool)

        for i, lt in enumerate(lane_tokens[:MAX_LANES]):
            try:
                centerline = nusc_map.discretize_lanes([lt], resolution_meters=2.0)
                if lt not in centerline or len(centerline[lt]) == 0:
                    continue
                pts = np.array(centerline[lt])[:, :2]  # (P, 2)
            except Exception:
                continue

            # Transform to ego frame
            pts_local = (R @ (pts - ego_pos).T).T  # (P, 2)

            # Resample to fixed number of points
            if len(pts_local) < 2:
                continue
            indices = np.linspace(0, len(pts_local) - 1, MAX_LANE_POINTS).astype(int)
            resampled = pts_local[indices]  # (MAX_LANE_POINTS, 2)

            # Flatten xy + lane type one-hot [driving, connector, other]
            flat_pts = resampled.flatten()  # (MAX_LANE_POINTS * 2,)

            # Simple lane type heuristic
            lane_type = np.array([1.0, 0.0, 0.0])  # assume driving lane

            lanes[i, :MAX_LANE_POINTS * 2] = flat_pts
            lanes[i, MAX_LANE_POINTS * 2:] = lane_type
            mask[i] = True

        return lanes, mask

    def __getitem__(self, idx: int) -> dict:
        sample_token = self.samples[idx]

        # --- Ego state (matching EgomotionData fields) ---
        ego = self._get_ego_pose(sample_token)
        ego_yaw = quaternion_to_yaw(ego["rotation"])

        # Compute ego velocity from previous sample's pose
        sample = self.nusc.get("sample", sample_token)
        vel_x, vel_y = 0.0, 0.0
        if sample["prev"]:
            prev_ego = self._get_ego_pose(sample["prev"])
            dt = 0.5  # 2Hz keyframe rate
            cos_y, sin_y = np.cos(-ego_yaw), np.sin(-ego_yaw)
            R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            dp = np.array(ego["translation"][:2]) - np.array(prev_ego["translation"][:2])
            local_vel = R @ dp / dt
            vel_x, vel_y = float(local_vel[0]), float(local_vel[1])

        ego_state = np.array([
            vel_x, vel_y, 0.0,     # velX, velY, velZ (body frame, m/s)
            0.0, 0.0, 0.0,        # angVelX, angVelY, angVelZ (placeholder)
            0.0, 0.0, 0.0,        # pos relative to self = origin
        ], dtype=np.float32)

        # --- Objects ---
        objects, obj_mask = self._get_objects(sample_token)

        # --- Lanes ---
        lanes, lane_mask = self._get_lanes(sample_token)

        # --- Future trajectory (ground truth, already interpolated to 10Hz) ---
        future_traj = self._get_future_trajectory(sample_token)

        # Handle edge case where trajectory is shorter than expected
        T = future_traj.shape[0]
        gt_waypoints = np.zeros((TRAJECTORY_STEPS, WAYPOINT_DIM), dtype=np.float32)
        wp_mask = np.zeros(TRAJECTORY_STEPS, dtype=bool)
        valid_steps = min(T, TRAJECTORY_STEPS)
        gt_waypoints[:valid_steps] = future_traj[:valid_steps]
        wp_mask[:valid_steps] = True

        # --- Navigation command ---
        nav_command = compute_nav_command(future_traj)

        # --- Meta-action label ---
        meta_action = compute_meta_action(future_traj)

        return {
            "ego_state": torch.from_numpy(ego_state),
            "objects": torch.from_numpy(objects),
            "object_mask": torch.from_numpy(obj_mask),
            "lanes": torch.from_numpy(lanes),
            "lane_mask": torch.from_numpy(lane_mask),
            "nav_command": torch.from_numpy(nav_command),
            "gt_waypoints": torch.from_numpy(gt_waypoints),
            "waypoint_mask": torch.from_numpy(wp_mask),
            "meta_action": torch.tensor(meta_action, dtype=torch.long),
        }
