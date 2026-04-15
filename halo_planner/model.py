"""
HALO Planner — Stage 1 trajectory planning network.

Custom transformer that consumes structured scene representations
(matching fusionengine::FusedPacket) and outputs future waypoints.

Architecture:
    SceneTokeniser → TransformerEncoder → TrajectoryDecoder → Waypoints

Designed to run on NVIDIA Drive AGX Orin via TensorRT export (FP16).
All operations are standard (Linear, MultiHeadAttention, LayerNorm, GELU)
for clean ONNX/TRT conversion.

Copyright (c) 2026 Halo Drive Ltd. All rights reserved.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants matching FusedPacket.hpp
# ---------------------------------------------------------------------------
MAX_OBJECTS = 64          # Padded max tracked objects per frame
MAX_LANES = 32            # Padded max lane polylines
MAX_LANE_POINTS = 20      # Points per lane polyline (resampled)
MAX_FREE_SPACE = 72       # Downsampled free space boundary points (from 360)
NUM_CLASSES = 3           # 0: Vehicle, 1: Pedestrian, 2: Cyclist

# Planner output
TRAJECTORY_STEPS = 40     # 40 waypoints @ 10Hz = 4.0 seconds
WAYPOINT_DIM = 4          # (x, y, heading, velocity)

# Meta-actions
META_ACTIONS = [
    "follow_lane",
    "lane_change_left",
    "lane_change_right",
    "stop",
    "yield",
    "reverse",
]
NUM_META_ACTIONS = len(META_ACTIONS)


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (for spatial positions)
# ---------------------------------------------------------------------------
class SinusoidalPosEncoding(nn.Module):
    """Encodes (x, y) positions into d-dimensional vectors using sin/cos."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D pos enc"
        self.half_d = d_model // 2

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: (B, N, 2) — x, y coordinates in metres
        Returns:
            (B, N, d_model) positional encoding
        """
        B, N, _ = xy.shape
        x = xy[..., 0:1]  # (B, N, 1)
        y = xy[..., 1:2]

        div_term = torch.exp(
            torch.arange(0, self.half_d, 2, device=xy.device, dtype=xy.dtype)
            * -(math.log(10000.0) / self.half_d)
        )  # (half_d // 2,)

        pe_x = torch.zeros(B, N, self.half_d, device=xy.device, dtype=xy.dtype)
        pe_y = torch.zeros(B, N, self.half_d, device=xy.device, dtype=xy.dtype)

        pe_x[..., 0::2] = torch.sin(x * div_term)
        pe_x[..., 1::2] = torch.cos(x * div_term)
        pe_y[..., 0::2] = torch.sin(y * div_term)
        pe_y[..., 1::2] = torch.cos(y * div_term)

        return torch.cat([pe_x, pe_y], dim=-1)  # (B, N, d_model)


# ---------------------------------------------------------------------------
# Scene Tokeniser
# Maps FusedPacket fields to a unified token sequence.
# ---------------------------------------------------------------------------
class SceneTokeniser(nn.Module):
    """
    Converts structured driving scene into a sequence of d_model-dim tokens.

    Token types:
        [EGO] — 1 token from EgomotionData (vel, ang_vel, ...)
        [OBJ] — N tokens from FusedDetection[] (pos, dims, class, conf, ...)
        [LANE] — M tokens from lane polylines (resampled control points)
        [NAV]  — 1 token from navigation command (route intent)

    Each token gets a type embedding added so the transformer knows
    what kind of entity it's attending to.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # --- Ego encoder ---
        # EgomotionData: velX/Y/Z (3) + angVelX/Y/Z (3) + pos x/y/z (3) = 9
        self.ego_mlp = nn.Sequential(
            nn.Linear(9, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Object encoder ---
        # FusedDetection: x/y/z (3) + w/l/h (3) + rotation (1) +
        #   classId one-hot (3) + fusedConfidence (1) + isDynamic (1) +
        #   hasLidar (1) + hasCamera (1) + velX/Y (2, from tracker) = 16
        self.obj_mlp = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Lane encoder ---
        # Each lane: MAX_LANE_POINTS * 2 (x,y per point) + lane_type one-hot (3)
        self.lane_mlp = nn.Sequential(
            nn.Linear(MAX_LANE_POINTS * 2 + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Navigation command encoder ---
        # One-hot: [left, straight, right] = 3
        self.nav_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Spatial position encoding ---
        self.pos_enc = SinusoidalPosEncoding(d_model)

        # --- Token type embeddings ---
        # 0=ego, 1=object, 2=lane, 3=nav
        self.type_embed = nn.Embedding(4, d_model)

    def forward(
        self,
        ego_state: torch.Tensor,       # (B, 9)
        objects: torch.Tensor,          # (B, N, 16)
        object_mask: torch.Tensor,      # (B, N) — True = valid
        lanes: torch.Tensor,            # (B, M, MAX_LANE_POINTS*2 + 3)
        lane_mask: torch.Tensor,        # (B, M) — True = valid
        nav_command: torch.Tensor,      # (B, 3) — one-hot [left, straight, right]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: (B, 1+N+M+1, d_model)
            key_padding_mask: (B, 1+N+M+1) — True = ignore (padding)
        """
        B = ego_state.shape[0]
        N = objects.shape[1]
        M = lanes.shape[1]
        device = ego_state.device

        # Encode each token type
        ego_tok = self.ego_mlp(ego_state).unsqueeze(1)          # (B, 1, d)
        obj_tok = self.obj_mlp(objects)                          # (B, N, d)
        lane_tok = self.lane_mlp(lanes)                          # (B, M, d)
        nav_tok = self.nav_mlp(nav_command).unsqueeze(1)         # (B, 1, d)

        # Add spatial position encoding to objects and lanes
        obj_xy = objects[..., :2]                                # (B, N, 2)
        obj_tok = obj_tok + self.pos_enc(obj_xy)

        lane_mid_idx = MAX_LANE_POINTS // 2
        lane_xy = lanes[..., lane_mid_idx*2 : lane_mid_idx*2+2] # midpoint xy
        lane_tok = lane_tok + self.pos_enc(lane_xy)

        # Add type embeddings
        ego_tok = ego_tok + self.type_embed(torch.zeros(B, 1, dtype=torch.long, device=device))
        obj_tok = obj_tok + self.type_embed(torch.ones(B, N, dtype=torch.long, device=device))
        lane_tok = lane_tok + self.type_embed(torch.full((B, M), 2, dtype=torch.long, device=device))
        nav_tok = nav_tok + self.type_embed(torch.full((B, 1), 3, dtype=torch.long, device=device))

        # Concatenate: [ego, objects, lanes, nav]
        tokens = torch.cat([ego_tok, obj_tok, lane_tok, nav_tok], dim=1)

        # Build key_padding_mask: True = ignore
        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        obj_pad = ~object_mask                                    # invert: True = pad
        lane_pad = ~lane_mask
        nav_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

        key_padding_mask = torch.cat([ego_mask, obj_pad, lane_pad, nav_mask], dim=1)

        return tokens, key_padding_mask


# ---------------------------------------------------------------------------
# Trajectory Decoder
# Learned queries → cross-attention to scene → MLP regression
# ---------------------------------------------------------------------------
class TrajectoryDecoder(nn.Module):
    """
    Decodes future trajectory from encoded scene representation.

    Uses T learned query embeddings (one per future timestep) that
    cross-attend to the scene tokens, then an MLP regresses
    (x, y, heading, velocity) per step.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        num_steps: int = TRAJECTORY_STEPS,
    ):
        super().__init__()
        self.num_steps = num_steps

        # Learned trajectory queries — one per future timestep
        self.traj_queries = nn.Embedding(num_steps, d_model)

        # Temporal position encoding for queries (fixed sinusoidal over time)
        self.register_buffer(
            "time_pos",
            self._make_time_pos(num_steps, d_model),
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Waypoint regression head: (x, y, heading, velocity)
        self.waypoint_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, WAYPOINT_DIM),
        )

        # Meta-action classification head
        self.meta_action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, NUM_META_ACTIONS),
        )

    @staticmethod
    def _make_time_pos(T: int, d: int) -> torch.Tensor:
        """Standard 1D sinusoidal encoding for timestep indices."""
        pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
        pe = torch.zeros(T, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, T, d)

    def forward(
        self,
        scene_tokens: torch.Tensor,     # (B, S, d_model)
        memory_mask: torch.Tensor,       # (B, S) — True = ignore
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            waypoints: (B, T, 4) — [x, y, heading, velocity] per step
            meta_logits: (B, NUM_META_ACTIONS)
        """
        B = scene_tokens.shape[0]
        device = scene_tokens.device

        # Build query: learned embedding + temporal position
        q_idx = torch.arange(self.num_steps, device=device)
        queries = self.traj_queries(q_idx).unsqueeze(0).expand(B, -1, -1)
        queries = queries + self.time_pos[:, :self.num_steps, :].to(device)

        # Decode: queries attend to scene tokens
        decoded = self.decoder(
            tgt=queries,
            memory=scene_tokens,
            memory_key_padding_mask=memory_mask,
        )  # (B, T, d_model)

        # Regress waypoints
        waypoints = self.waypoint_head(decoded)  # (B, T, 4)

        # Meta-action: pool all decoded tokens
        pooled = decoded.mean(dim=1)             # (B, d_model)
        meta_logits = self.meta_action_head(pooled)  # (B, NUM_META_ACTIONS)

        return waypoints, meta_logits


# ---------------------------------------------------------------------------
# Full Planner Model
# ---------------------------------------------------------------------------
class HaloPlanner(nn.Module):
    """
    HALO Planner Stage 1 — complete model.

    Input:  Structured scene from FusedPacket (ego, objects, lanes, nav)
    Output: Future trajectory (40 waypoints) + meta-action classification

    Architecture:
        SceneTokeniser (MLPs + pos enc)
        → TransformerEncoder (6 layers, 8 heads, d=256)
        → TrajectoryDecoder (3 layers cross-attention + MLP head)

    ~25M parameters. Target: <10ms inference on Orin (FP16 TensorRT).
    """

    def __init__(
        self,
        d_model: int = 256,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        num_steps: int = TRAJECTORY_STEPS,
    ):
        super().__init__()

        self.tokeniser = SceneTokeniser(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        self.decoder = TrajectoryDecoder(
            d_model=d_model,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            num_steps=num_steps,
        )

    def forward(
        self,
        ego_state: torch.Tensor,
        objects: torch.Tensor,
        object_mask: torch.Tensor,
        lanes: torch.Tensor,
        lane_mask: torch.Tensor,
        nav_command: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with:
            "waypoints":    (B, T, 4) — [x, y, heading, velocity]
            "meta_logits":  (B, NUM_META_ACTIONS)
        """
        # Tokenise scene
        tokens, key_mask = self.tokeniser(
            ego_state, objects, object_mask, lanes, lane_mask, nav_command
        )

        # Encode scene (self-attention across all entities)
        encoded = self.encoder(tokens, src_key_padding_mask=key_mask)

        # Decode trajectory
        waypoints, meta_logits = self.decoder(encoded, key_mask)

        return {
            "waypoints": waypoints,
            "meta_logits": meta_logits,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B = 4
    N = MAX_OBJECTS
    M = MAX_LANES

    model = HaloPlanner()
    print(f"HALO Planner — {model.count_parameters() / 1e6:.1f}M parameters")

    # Dummy inputs matching FusedPacket dimensions
    ego = torch.randn(B, 9)
    objs = torch.randn(B, N, 16)
    obj_mask = torch.ones(B, N, dtype=torch.bool)
    obj_mask[:, 30:] = False  # Only 30 real objects
    lns = torch.randn(B, M, MAX_LANE_POINTS * 2 + 3)
    ln_mask = torch.ones(B, M, dtype=torch.bool)
    ln_mask[:, 6:] = False    # Only 6 real lanes
    nav = torch.zeros(B, 3)
    nav[:, 1] = 1.0           # "straight"

    out = model(ego, objs, obj_mask, lns, ln_mask, nav)
    print(f"Waypoints shape: {out['waypoints'].shape}")    # (4, 40, 4)
    print(f"Meta-action logits: {out['meta_logits'].shape}")  # (4, 6)
    print("Sanity check passed.")
