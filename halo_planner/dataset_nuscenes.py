"""
HALO Planner — nuScenes dataset adapter.

Converts nuScenes annotations into the structured scene format
matching FusedPacket.hpp for training the planner.

UPDATED: 
 - Now extracts true dynamic object velocities using nusc.box_velocity.
 - Applies Trajectory Perturbation (Covariate Shift fix) during training.
 - Calculates true ego angular velocity.
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

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def quaternion_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from quaternion [w, x, y, z]."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def compute_nav_command(future_traj: np.ndarray) -> np.ndarray:
    """Derive navigation command from future trajectory."""
    if len(future_traj) < 2:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    lateral = future_traj[-1, 1] - future_traj[0, 1]
    if lateral > 2.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif lateral < -2.0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

def compute_meta_action(future_traj: np.ndarray) -> int:
    """Derive meta-action label from future trajectory."""
    if len(future_traj) < 2:
        return 0  # follow_lane

    elapsed = len(future_traj) * 0.1
    speed = np.linalg.norm(future_traj[-1, :2] - future_traj[0, :2]) / max(elapsed, 1e-3)
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
        self.split = split
        self.future_secs = future_secs
        self.past_secs = past_secs
        self.annotation_rate = 2.0       
        self.output_rate = 10.0          
        self.future_steps_ann = int(future_secs * self.annotation_rate)

        splits = create_splits_scenes()
        split_scenes = splits[split] if split in splits else splits["mini_train"]

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
            if len(scene_samples) > self.future_steps_ann:
                self.samples.extend(scene_samples[:-self.future_steps_ann])

        self.map_cache = {}
        for log in self.nusc.log:
            loc = log["location"]
            if loc not in self.map_cache:
                self.map_cache[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)

        print(f"[NuScenesDataset] Loaded {len(self.samples)} samples ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_ego_pose(self, sample_token: str) -> dict:
        sample = self.nusc.get("sample", sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        return ego

    def _get_future_trajectory(self, sample_token: str, base_pos: np.ndarray, base_yaw: float) -> np.ndarray:
        cos_y, sin_y = np.cos(-base_yaw), np.sin(-base_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        keyframes = []  
        token = sample_token
        dt_ann = 1.0 / self.annotation_rate 
        
        for i in range(self.future_steps_ann):
            sample = self.nusc.get("sample", token)
            if not sample["next"]:
                break
            token = sample["next"]
            fut_ego = self._get_ego_pose(token)
            fut_pos = np.array(fut_ego["translation"][:2])
            fut_yaw = quaternion_to_yaw(fut_ego["rotation"])

            # Transform into the (potentially perturbed) base frame
            local_pos = R @ (fut_pos - base_pos)
            local_heading = _wrap_angle(fut_yaw - base_yaw)
            t = (i + 1) * dt_ann

            keyframes.append([t, local_pos[0], local_pos[1], local_heading])

        if len(keyframes) < 2:
            return np.zeros((1, 4), dtype=np.float32)

        keyframes = np.array(keyframes, dtype=np.float32)

        # Speed calculation (independent of reference frame rotation)
        K = len(keyframes)
        kf_t = keyframes[:, 0]
        kf_xy = keyframes[:, 1:3]
        kf_speed = np.zeros(K, dtype=np.float32)
        kf_speed[0] = np.linalg.norm(kf_xy[0]) / max(kf_t[0], 1e-3)
        for i in range(1, K):
            d = np.linalg.norm(kf_xy[i] - kf_xy[i - 1])
            dt_kf = max(kf_t[i] - kf_t[i - 1], 1e-3)
            kf_speed[i] = d / dt_kf

        dt_out = 1.0 / self.output_rate
        t_out = np.arange(1, TRAJECTORY_STEPS + 1) * dt_out 
        t_out = np.clip(t_out, 0, keyframes[-1, 0]) 

        x_interp = np.interp(t_out, keyframes[:, 0], keyframes[:, 1])
        y_interp = np.interp(t_out, keyframes[:, 0], keyframes[:, 2])
        hdg_interp = np.interp(t_out, keyframes[:, 0], keyframes[:, 3])
        vel = np.interp(t_out, kf_t, kf_speed).astype(np.float32)

        waypoints = np.stack([x_interp, y_interp, hdg_interp, vel], axis=-1)
        return waypoints.astype(np.float32)

    def _get_objects(self, sample_token: str, base_pos: np.ndarray, base_yaw: float) -> tuple[np.ndarray, np.ndarray]:
        sample = self.nusc.get("sample", sample_token)
        
        cos_y, sin_y = np.cos(-base_yaw), np.sin(-base_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        objects = np.zeros((MAX_OBJECTS, 16), dtype=np.float32)
        mask = np.zeros(MAX_OBJECTS, dtype=bool)

        count = 0
        for ann_token in sample["anns"]:
            if count >= MAX_OBJECTS:
                break
            ann = self.nusc.get("sample_annotation", ann_token)
            
            class_id = None
            for key, cid in NUSCENES_CLASS_MAP.items():
                if key in ann["category_name"]:
                    class_id = cid
                    break
            if class_id is None:
                continue

            world_pos = np.array(ann["translation"][:2])
            local_pos = R @ (world_pos - base_pos)

            if np.linalg.norm(local_pos) > 80.0:
                continue

            # Ego z logic is approximate here to keep signature clean
            z = ann["translation"][2] 
            yaw = _wrap_angle(quaternion_to_yaw(ann["rotation"]) - base_yaw)
            w, l, h = ann["size"] 

            class_onehot = np.zeros(3, dtype=np.float32)
            class_onehot[class_id] = 1.0

            # NEW: Extract true velocity using nuScenes API
            world_vel = self.nusc.box_velocity(ann_token)
            if np.any(np.isnan(world_vel)):
                vel_x, vel_y = 0.0, 0.0
            else:
                local_vel = R @ world_vel[:2]
                vel_x, vel_y = float(local_vel[0]), float(local_vel[1])

            objects[count] = [
                local_pos[0], local_pos[1], z,
                w, l, h,
                yaw,
                class_onehot[0], class_onehot[1], class_onehot[2],
                1.0, 1.0, 1.0, 1.0,
                vel_x, vel_y,  # <--- True velocities injected!
            ]
            mask[count] = True
            count += 1

        return objects, mask

    def _get_lanes(self, sample_token: str, base_pos: np.ndarray, base_yaw: float) -> tuple[np.ndarray, np.ndarray]:
        sample = self.nusc.get("sample", sample_token)
        log = self.nusc.get("log", self.nusc.get("scene", sample["scene_token"])["log_token"])
        nusc_map = self.map_cache[log["location"]]

        cos_y, sin_y = np.cos(-base_yaw), np.sin(-base_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        radius = 50.0  
        lane_records = nusc_map.get_records_in_radius(base_pos[0], base_pos[1], radius, ["lane", "lane_connector"])
        lane_tokens = lane_records.get("lane", []) + lane_records.get("lane_connector", [])

        lanes = np.zeros((MAX_LANES, MAX_LANE_POINTS * 2 + 3), dtype=np.float32)
        mask = np.zeros(MAX_LANES, dtype=bool)

        for i, lt in enumerate(lane_tokens[:MAX_LANES]):
            try:
                centerline = nusc_map.discretize_lanes([lt], resolution_meters=2.0)
                if lt not in centerline or len(centerline[lt]) == 0:
                    continue
                pts = np.array(centerline[lt])[:, :2] 
            except Exception:
                continue

            pts_local = (R @ (pts - base_pos).T).T  

            if len(pts_local) < 2:
                continue
            indices = np.linspace(0, len(pts_local) - 1, MAX_LANE_POINTS).astype(int)
            resampled = pts_local[indices]  

            flat_pts = resampled.flatten()  
            lane_type = np.array([1.0, 0.0, 0.0])  

            lanes[i, :MAX_LANE_POINTS * 2] = flat_pts
            lanes[i, MAX_LANE_POINTS * 2:] = lane_type
            mask[i] = True

        return lanes, mask

    def __getitem__(self, idx: int) -> dict:
        sample_token = self.samples[idx]

        # 1. Fetch true ground truth ego state
        ego = self._get_ego_pose(sample_token)
        true_yaw = quaternion_to_yaw(ego["rotation"])
        true_pos = np.array(ego["translation"][:2])

        # 2. Covariate Shift Fix: Trajectory Perturbation
        noise_x, noise_y, noise_yaw = 0.0, 0.0, 0.0
        
        # Only inject noise during training to force the model to learn recovery
        if self.split == "train" and np.random.rand() < 0.5:
            noise_x = np.random.normal(0, 0.1)     # Slight surge/lag
            noise_y = np.random.normal(0, 0.4)     # Drift up to ~1m laterally
            noise_yaw = np.random.normal(0, 0.05)  # Slight heading error

        # Transform local noise to world coordinates
        cos_y, sin_y = np.cos(true_yaw), np.sin(true_yaw)
        world_noise_x = cos_y * noise_x - sin_y * noise_y
        world_noise_y = sin_y * noise_x + cos_y * noise_y
        
        base_pos = true_pos + np.array([world_noise_x, world_noise_y])
        base_yaw = true_yaw + noise_yaw

        # 3. Calculate true Ego velocities 
        sample = self.nusc.get("sample", sample_token)
        vel_x, vel_y, ang_vel_z = 0.0, 0.0, 0.0
        
        if sample["prev"]:
            prev_ego = self._get_ego_pose(sample["prev"])
            prev_yaw = quaternion_to_yaw(prev_ego["rotation"])
            dt = 0.5  
            
            # Rotation matrix for the *base* (potentially noisy) frame
            cos_b, sin_b = np.cos(-base_yaw), np.sin(-base_yaw)
            R_base = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
            
            dp = np.array(ego["translation"][:2]) - np.array(prev_ego["translation"][:2])
            local_vel = R_base @ (dp / dt)
            vel_x, vel_y = float(local_vel[0]), float(local_vel[1])
            
            # True angular velocity
            ang_vel_z = _wrap_angle(true_yaw - prev_yaw) / dt

        ego_state = np.array([
            vel_x, vel_y, 0.0,          # velX, velY, velZ 
            0.0, 0.0, ang_vel_z,        # angVelX, angVelY, angVelZ 
            0.0, 0.0, 0.0,              # pos relative to self
        ], dtype=np.float32)

        # 4. Extract environment relative to the perturbed base frame
        objects, obj_mask = self._get_objects(sample_token, base_pos, base_yaw)
        lanes, lane_mask = self._get_lanes(sample_token, base_pos, base_yaw)
        future_traj = self._get_future_trajectory(sample_token, base_pos, base_yaw)

        T = future_traj.shape[0]
        gt_waypoints = np.zeros((TRAJECTORY_STEPS, WAYPOINT_DIM), dtype=np.float32)
        wp_mask = np.zeros(TRAJECTORY_STEPS, dtype=bool)
        valid_steps = min(T, TRAJECTORY_STEPS)
        gt_waypoints[:valid_steps] = future_traj[:valid_steps]
        wp_mask[:valid_steps] = True

        nav_command = compute_nav_command(future_traj)
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