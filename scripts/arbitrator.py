import math
import numpy as np

class TrajectoryArbitrator:
    """
    Middle layer between the Neural Planner and Kinematic Controller.
    Responsible for:
      1. State Diagnostics (Cross-Track Error)
      2. Temporal Coherency (Trajectory Smoothing/Stitching)
      3. Deterministic Safety (TTC / AEB Overrides)
    """
    
    def __init__(self, ttc_threshold: float = 1.5, lateral_jump_limit: float = 0.5):
        self.ttc_threshold = ttc_threshold
        self.lateral_jump_limit = lateral_jump_limit
        
        self.prev_waypoints = None
        self.last_cte = 0.0

    def calculate_cte(self, ego_pos, waypoints) -> float:
        """Calculates Cross-Track Error (CTE) from the ego to the trajectory."""
        if waypoints is None:
            return 0.0
        # Simple distance to the immediate intended path (waypoints 0 to 5)
        path_segment = waypoints[:5, :2] 
        distances = np.linalg.norm(path_segment - ego_pos[:2], axis=1)
        return float(np.min(distances))

    def check_ttc_safety(self, agent) -> bool:
        """
        Calculates Time-to-Collision (TTC) for all spawned objects.
        Returns True if an emergency stop is required.
        """
        if not hasattr(agent, "engine") or not hasattr(agent.engine, "traffic_manager"):
            return False
            
        ego_pos = np.array(agent.position)
        ego_vel = np.array(agent.velocity)
        ego_speed = np.linalg.norm(ego_vel)
        
        if ego_speed < 0.5:
            return False  # Already stopped
            
        tm = agent.engine.traffic_manager
        for vid, vehicle in tm.spawned_objects.items():
            if vehicle is agent:
                continue
                
            obj_pos = np.array(vehicle.position)
            obj_vel = np.array(vehicle.velocity)
            
            # Relative kinematics
            rel_pos = obj_pos - ego_pos
            dist = np.linalg.norm(rel_pos)
            
            if dist > 40.0:
                continue # Ignore distant objects
                
            # Project velocities along the line of sight
            los_dir = rel_pos / max(dist, 1e-3)
            ego_v_los = np.dot(ego_vel, los_dir)
            obj_v_los = np.dot(obj_vel, los_dir)
            
            rel_v_closing = ego_v_los - obj_v_los
            
            # If closing in fast enough, check TTC
            if rel_v_closing > 0.5:
                ttc = dist / rel_v_closing
                if ttc < self.ttc_threshold:
                    return True # Emergency!
                    
        return False

    def arbitrate(self, raw_waypoints: np.ndarray, raw_meta: np.ndarray, agent) -> dict:
        """
        Ingests the raw neural network output, diagnoses the state, 
        and returns a structurally sound, safe plan for the controller.
        """
        # 1. State Diagnostics (CTE)
        # Ego is always at (0,0) in the planner's local frame.
        self.last_cte = self.calculate_cte(np.array([0.0, 0.0]), raw_waypoints)
        
        arbitrated_wp = raw_waypoints.copy()
        
        # 2. Temporal Coherency (Smoothing)
        if self.prev_waypoints is not None:
            # Check for massive lateral hallucinations at WP 10 (1.0s ahead)
            lateral_jump = abs(raw_waypoints[10, 1] - self.prev_waypoints[10, 1])
            
            if lateral_jump > self.lateral_jump_limit:
                # The model hallucinated! Reject the sudden jump and blend 
                # heavily towards the previous safe trajectory (EMA).
                alpha = 0.2 # 20% new, 80% old
                arbitrated_wp[:, :2] = (alpha * raw_waypoints[:, :2]) + ((1 - alpha) * self.prev_waypoints[:, :2])
            else:
                # Normal temporal smoothing
                alpha = 0.6
                arbitrated_wp[:, :2] = (alpha * raw_waypoints[:, :2]) + ((1 - alpha) * self.prev_waypoints[:, :2])
                
        self.prev_waypoints = arbitrated_wp
        
        # 3. Deterministic Safety (TTC Override)
        emergency_brake = self.check_ttc_safety(agent)
        if emergency_brake:
            # Override the velocity profile to 0.0
            arbitrated_wp[:, 3] = 0.0
            # Override meta-action to STOP
            raw_meta *= 0.0 
            raw_meta[3] = 10.0 # Force STOP index to highest logit
            
        return {
            "waypoints": arbitrated_wp,
            "meta_logits": raw_meta,
            "cte": self.last_cte,
            "aeb_triggered": emergency_brake
        }