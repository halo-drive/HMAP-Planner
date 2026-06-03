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
        w_lane: float = 0.1,
        lane_band: float = 1.75,
    ):
        super().__init__()
        self.w_traj = w_traj
        self.w_heading = w_heading
        self.w_velocity = w_velocity
        self.w_meta = w_meta
        self.w_coll = w_coll
        self.w_lane = w_lane
        # Soft lane-keeping band (m): no penalty within this lateral distance of
        # the nearest lane centerline; linear penalty beyond. ~half a 3.5m lane,
        # so normal within-lane variation isn't penalised, only genuine departure.
        self.lane_band = lane_band
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
        lane_points: torch.Tensor | None = None,         # (B, M, MAX_LANE_POINTS*2) for lane-keeping
        lane_mask: torch.Tensor | None = None,           # (B, M)
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

        # --- Collision penalty  ---
        l_coll = torch.tensor(0.0, device=pred_xy.device)
        if object_positions is not None and object_mask is not None:
            # Distance from each predicted waypoint to nearest object
            # pred_xy: (B, T, 2), obj_pos: (B, N, 2)
            diff = pred_xy.unsqueeze(2) - object_positions.unsqueeze(1)  #(B, T, N, 2)
            dist = diff.norm(dim=-1)  #(B, T, N)

            # Mask out invalid objects
            dist = dist.masked_fill(~object_mask.unsqueeze(1), float("inf"))

            # Min distance per waypoint
            min_dist = dist.min(dim=2).values  #(B, T)

            # Penalise waypoints closer than 2m to any object
            collision_margin = 2.0
            violation = F.relu(collision_margin - min_dist)  # 0 if far, positive if close
            l_coll = (violation * waypoint_mask).sum() / waypoint_mask.sum().clamp(min=1)

        # --- Lane-keeping penalty  ---
        # Penalise predicted waypoints that stray far from the nearest lane
        # centerline point. Soft hinge: zero within `lane_band` metres, linear
        # beyond. Distance is to the nearest of ALL valid lane points (no lane
        # identity available), a good proxy on open road. Small weight so it
        # shapes behaviour without overriding imitation.
        l_lane = torch.tensor(0.0, device=pred_xy.device)
        if lane_points is not None and lane_mask is not None:
            B, T, _ = pred_xy.shape
            M = lane_points.shape[1]
            # (B, M, MAX_LANE_POINTS*2) -> (B, M, P, 2) -> (B, M*P, 2)
            lp = lane_points.view(B, M, -1, 2)
            P = lp.shape[2]
            lp_flat = lp.reshape(B, M * P, 2)                       # (B, M*P, 2)
            # Per-point validity: a lane's points are valid iff its mask is True.
            pt_valid = lane_mask.unsqueeze(-1).expand(B, M, P).reshape(B, M * P)  # (B, M*P)

            # Distance from each predicted waypoint to every lane point.
            # pred_xy: (B, T, 2) -> (B, T, 1, 2); lp_flat: (B, 1, M*P, 2)
            d = (pred_xy.unsqueeze(2) - lp_flat.unsqueeze(1)).norm(dim=-1)  # (B, T, M*P)
            # Invalidate padded lane points.
            d = d.masked_fill(~pt_valid.unsqueeze(1), float("inf"))
            min_d = d.min(dim=2).values                              # (B, T)

            # Samples with NO valid lanes produce inf -> exclude them from the term.
            has_lane = torch.isfinite(min_d)                         # (B, T)
            valid = waypoint_mask.bool() & has_lane
            # Soft hinge beyond the band.
            departure = F.relu(min_d - self.lane_band)
            departure = torch.where(valid, departure, torch.zeros_like(departure))
            denom = valid.float().sum().clamp(min=1)
            l_lane = departure.sum() / denom

        # --- Total ---
        total = (
            self.w_traj * l_traj
            + self.w_heading * l_heading
            + self.w_velocity * l_velocity
            + self.w_meta * l_meta
            + self.w_coll * l_coll
            + self.w_lane * l_lane
        )

        return {
            "total": total,
            "traj": l_traj,
            "heading": l_heading,
            "velocity": l_velocity,
            "meta": l_meta,
            "collision": l_coll,
            "lane": l_lane,
        }