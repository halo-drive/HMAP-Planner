"""
HALO Planner — Training losses.

Combines trajectory regression loss with meta-action classification
and optional collision penalty for safety-aware planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlannerLoss(nn.Module):
    """
    Multi-task loss for trajectory planning.

    Components:
        L_traj:     Smooth L1 on (x, y) waypoint positions
        L_heading:  Smooth L1 on heading angle (wrapped)
        L_velocity: Smooth L1 on velocity
        L_meta:     Cross-entropy on meta-action classification
        L_coll:     Optional collision penalty (distance to nearest object)

    Total = w_traj * L_traj + w_hdg * L_heading + w_vel * L_velocity
          + w_meta * L_meta + w_coll * L_coll
    """

    def __init__(
        self,
        w_traj: float = 1.0,
        w_heading: float = 0.5,
        w_velocity: float = 0.2,
        w_meta: float = 0.5,
        w_coll: float = 1.0,
    ):
        super().__init__()
        self.w_traj = w_traj
        self.w_heading = w_heading
        self.w_velocity = w_velocity
        self.w_meta = w_meta
        self.w_coll = w_coll
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean", beta=1.0)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        """Wrap angle difference to [-pi, pi]."""
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(
        self,
        pred_waypoints: torch.Tensor,    # (B, T, 4) — [x, y, heading, vel]
        gt_waypoints: torch.Tensor,       # (B, T, 4)
        pred_meta: torch.Tensor,          # (B, NUM_META_ACTIONS)
        gt_meta: torch.Tensor,            # (B,) — class index
        waypoint_mask: torch.Tensor,      # (B, T) — True = valid timestep
        object_positions: torch.Tensor | None = None,   # (B, N, 2) for collision
        object_mask: torch.Tensor | None = None,         # (B, N)
    ) -> dict[str, torch.Tensor]:

        # --- Position loss (x, y) ---
        pred_xy = pred_waypoints[..., :2]       # (B, T, 2)
        gt_xy = gt_waypoints[..., :2]
        pos_err = F.smooth_l1_loss(pred_xy, gt_xy, reduction="none")  # (B, T, 2)
        pos_err = (pos_err * waypoint_mask.unsqueeze(-1)).sum() / waypoint_mask.sum().clamp(min=1)
        l_traj = pos_err

        # --- Heading loss (angle-wrapped) ---
        pred_hdg = pred_waypoints[..., 2]
        gt_hdg = gt_waypoints[..., 2]
        hdg_diff = self._wrap_angle(pred_hdg - gt_hdg).abs()
        hdg_err = (hdg_diff * waypoint_mask).sum() / waypoint_mask.sum().clamp(min=1)
        l_heading = hdg_err

        # --- Velocity loss ---
        pred_vel = pred_waypoints[..., 3]
        gt_vel = gt_waypoints[..., 3]
        vel_err = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
        vel_err = (vel_err * waypoint_mask).sum() / waypoint_mask.sum().clamp(min=1)
        l_velocity = vel_err

        # --- Meta-action classification ---
        l_meta = F.cross_entropy(pred_meta, gt_meta)

        # --- Collision penalty (optional) ---
        l_coll = torch.tensor(0.0, device=pred_xy.device)
        if object_positions is not None and object_mask is not None:
            # Distance from each predicted waypoint to nearest object
            # pred_xy: (B, T, 2), obj_pos: (B, N, 2)
            diff = pred_xy.unsqueeze(2) - object_positions.unsqueeze(1)  # (B, T, N, 2)
            dist = diff.norm(dim=-1)  # (B, T, N)

            # Mask out invalid objects
            dist = dist.masked_fill(~object_mask.unsqueeze(1), float("inf"))

            # Min distance per waypoint
            min_dist = dist.min(dim=2).values  # (B, T)

            # Penalise waypoints closer than 2m to any object
            collision_margin = 2.0
            violation = F.relu(collision_margin - min_dist)  # 0 if far, positive if close
            l_coll = (violation * waypoint_mask).sum() / waypoint_mask.sum().clamp(min=1)

        # --- Total ---
        total = (
            self.w_traj * l_traj
            + self.w_heading * l_heading
            + self.w_velocity * l_velocity
            + self.w_meta * l_meta
            + self.w_coll * l_coll
        )

        return {
            "total": total,
            "traj": l_traj,
            "heading": l_heading,
            "velocity": l_velocity,
            "meta": l_meta,
            "collision": l_coll,
        }
