# HALO Planner

Internall Trajectory planning module for the HALO Drive autonomous driving stack. A custom transformer-based planner that consumes structured perception data and outputs drivable trajectories — no raw camera frames, no pretrained LLM weights.

Built at [Halo Drive Ltd], Glasgow.

## Architecture

```
FusedPacket (Fusion Engine)
    ├── Ego state (9)         ─┐
    ├── Objects (64 × 16)      │── SceneTokeniser ── TransformerEncoder (6L) ── TrajectoryDecoder (3L)
    ├── Lanes (32 × 43)        │         per-type MLPs        8-head self-attn       cross-attn + MLP
    └── Nav command (3)       ─┘              ↓                     ↓                      ↓
                                         98 tokens @ 256d    contextualised tokens    40 waypoints (x,y,θ,v)
                                                                                     + meta-action class
```

**Stage 1 (this repo):** 8.3M parameter transformer planner. Consumes structured scene data matching `FusedPacket.hpp`. Outputs 40 waypoints at 10Hz (4.0s horizon) plus a meta-action classification (follow lane, lane change, stop, yield, reverse).

**Stage 2 (planned):** VLA reasoning module using Alpamayo-distilled model for trajectory validation with Chain-of-Causation reasoning. Parked until Stage 1 is integrated with the Fusion Engine.

## Results

| Metric | Value |
|--------|-------|
| Best val ADE | 1.677m (epoch 20) |
| Best val FDE | 3.869m |
| Parameters | 8,298,694 (8.3M) |
| Inference | 10-17ms on RTX 3060, 57-60 Hz |
| Training | 50 epochs, ~19h on single GPU |
| Dataset | nuScenes v1.0-trainval (22,530 train / 4,819 val) |

**MetaDrive closed-loop (straights):** 98.6m driven, 49.3% route completion, 0 collisions, 60 Hz inference.

## Project structure

```
halo_planner/
├── halo_planner/
│   ├── model.py              # HaloPlanner network (SceneTokeniser + Encoder + Decoder)
│   ├── losses.py             # Multi-component loss (traj + heading + velocity + meta + collision)
│   └── dataset_nuscenes.py   # nuScenes → FusedPacket adapter (2Hz→10Hz interpolation)
├── scripts/
│   ├── train.py              # Training loop (tqdm, TensorBoard, resume, FDE tracking)
│   ├── verify_dataset.py     # 7-check pre-training verification
│   └── test_metadrive.py     # Closed-loop evaluation in MetaDrive simulator
├── checkpoints/
│   ├── best_planner.pt       # Best model (epoch 20, ADE 1.677m)
│   ├── latest_planner.pt     # Latest checkpoint
│   └── train_history.json    # Full training history
└── data/
    └── nuscenes/             # Symlinks to nuScenes v1.0-trainval on SSD
```

## Input format

The planner consumes the same structured data that the Fusion Engine produces at runtime. During training, nuScenes ground-truth annotations are mapped to this format.

**Objects (16 floats per detection):** `x, y, z, width, length, height, rotation, class_onehot[3], fusedConfidence, isDynamic, hasLidarSource, hasCameraSource, velX, velY` — matches `FusedDetection` in `FusedPacket.hpp`.

**Ego state (9 floats):** `velX, velY, velZ, angVelX, angVelY, angVelZ, posX, posY, posZ` — matches `EgomotionData`.

**Lanes (43 floats per lane):** 20 centerline points × 2 coords + 3 lane-type one-hot — from Lanelet2 at runtime, nuScenes map expansion during training.

**Nav command (3 floats):** One-hot `[left, straight, right]`.

## Training

```bash
# Setup
cd ~/halo_planner
python -m venv planner_env && source planner_env/bin/activate
# please note that environment can be recreated using the requirements.txt
pip install torch nuscenes-devkit tensorboard tqdm

# Verify dataset
python scripts/verify_dataset.py --dataroot data/nuscenes --version v1.0-trainval

# data directory isnt available and the linker can be provided to Internall requests raised only.

# Train
python scripts/train.py \
    --dataroot data/nuscenes \
    --version v1.0-trainval \
    --epochs 50 --batch_size 16 --lr 1e-4

# Monitor
tensorboard --logdir checkpoints/tb_logs --port 6006
```

## MetaDrive testing

```bash
pip install -e metadrive/
# this is a third party repository used, for simulation using ScenarioNet for different scenarios, however, you are not allowed clone it separetly, you may use the submodule feature to init metadrive.

# 3D rendered window
python scripts/test_metadrive.py \
    --checkpoint checkpoints/best_planner.pt \
    --render --map SSS --traffic_density 0.2

# Headless with metrics
python scripts/test_metadrive.py \
    --checkpoint checkpoints/best_planner.pt \
    --map SCCS --num_scenarios 10
```

Map codes: `S` straight, `C` curve, `X` intersection, `O` roundabout, `T` T-junction.

## Deployment target

NVIDIA Drive AGX Orin via TensorRT. The planner sits between the Fusion Engine (IPC socket input) and the HMAP Controller_Wrapper (CAN bus output) in the vehicle stack:

```
Cameras/LiDAR → Perception Servers → Fusion Engine → [HALO Planner] → HMAP Controller → CAN bus
                 (driveseg, lidardet)   (FusedPacket)                    (steer/throttle/brake)
```

## References

- [UniAD](https://arxiv.org/abs/2212.10156) — architectural pattern (CVPR 2023 Best Paper)
- [ChauffeurNet](https://arxiv.org/abs/1812.03079) — mid-level representation imitation learning (Waymo 2018)
- [Alpamayo-R1](https://arxiv.org/abs/2511.00088) — Stage 2 VLA reference (NVIDIA)
- [NAVSIM](https://arxiv.org/abs/2406.15349) — evaluation benchmark (NeurIPS 2024)

## License

Proprietary — Halo Drive Ltd. All rights reserved. 2026
## Author -
@sudoashwin
