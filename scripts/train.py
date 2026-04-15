"""
HALO Planner — Training script.

Usage:
    python scripts/train.py --dataroot /data/nuscenes --version v1.0-mini --epochs 50

For full training:
    python scripts/train.py --dataroot /data/nuscenes --version v1.0-trainval --epochs 100 --batch_size 32
"""

import argparse
import os
import sys
import time
import json
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halo_planner.model import HaloPlanner
from halo_planner.losses import PlannerLoss
from halo_planner.dataset_nuscenes import NuScenesDataset


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def train_one_epoch(model, loader, criterion, optimiser, device, epoch, writer, global_step):
    model.train()
    total_loss = 0.0
    loss_accum = {"traj": 0.0, "heading": 0.0, "velocity": 0.0, "meta": 0.0, "collision": 0.0}
    ade_accum = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"  Train {epoch:>3d}",
        unit="batch",
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=True,
    )

    for batch_idx, batch in enumerate(pbar):
        ego = batch["ego_state"].to(device)
        objs = batch["objects"].to(device)
        obj_mask = batch["object_mask"].to(device)
        lanes = batch["lanes"].to(device)
        lane_mask = batch["lane_mask"].to(device)
        nav = batch["nav_command"].to(device)
        gt_wp = batch["gt_waypoints"].to(device)
        wp_mask = batch["waypoint_mask"].to(device)
        meta = batch["meta_action"].to(device)

        out = model(ego, objs, obj_mask, lanes, lane_mask, nav)

        losses = criterion(
            pred_waypoints=out["waypoints"],
            gt_waypoints=gt_wp,
            pred_meta=out["meta_logits"],
            gt_meta=meta,
            waypoint_mask=wp_mask,
            object_positions=objs[..., :2],
            object_mask=obj_mask,
        )

        optimiser.zero_grad()
        losses["total"].backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item()
        optimiser.step()

        # Compute batch ADE for progress bar
        with torch.no_grad():
            pred_xy = out["waypoints"][..., :2]
            gt_xy = gt_wp[..., :2]
            displacement = (pred_xy - gt_xy).norm(dim=-1)
            batch_ade = (displacement * wp_mask).sum() / wp_mask.sum().clamp(min=1)
            ade_accum += batch_ade.item()

        total_loss += losses["total"].item()
        for k in loss_accum:
            loss_accum[k] += losses[k].item()
        n_batches += 1
        global_step += 1

        # Update progress bar
        avg_loss = total_loss / n_batches
        avg_ade = ade_accum / n_batches
        pbar.set_postfix(
            loss=f"{avg_loss:.3f}",
            ADE=f"{avg_ade:.2f}m",
            traj=f"{losses['traj'].item():.3f}",
            grad=f"{grad_norm:.1f}",
        )

        # TensorBoard per-step logging (every 50 batches to avoid overhead)
        if global_step % 50 == 0:
            writer.add_scalar("batch/loss", losses["total"].item(), global_step)
            writer.add_scalar("batch/ADE", batch_ade.item(), global_step)
            writer.add_scalar("batch/grad_norm", grad_norm, global_step)
            for k in loss_accum:
                writer.add_scalar(f"batch/{k}", losses[k].item(), global_step)

    pbar.close()

    avg_loss = total_loss / max(n_batches, 1)
    avg_ade = ade_accum / max(n_batches, 1)
    for k in loss_accum:
        loss_accum[k] /= max(n_batches, 1)
    return avg_loss, loss_accum, avg_ade, global_step


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    ade_sum = 0.0
    fde_sum = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"  Val   {epoch:>3d}",
        unit="batch",
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=True,
    )

    for batch in pbar:
        ego = batch["ego_state"].to(device)
        objs = batch["objects"].to(device)
        obj_mask = batch["object_mask"].to(device)
        lanes = batch["lanes"].to(device)
        lane_mask = batch["lane_mask"].to(device)
        nav = batch["nav_command"].to(device)
        gt_wp = batch["gt_waypoints"].to(device)
        wp_mask = batch["waypoint_mask"].to(device)
        meta = batch["meta_action"].to(device)

        out = model(ego, objs, obj_mask, lanes, lane_mask, nav)

        losses = criterion(
            pred_waypoints=out["waypoints"],
            gt_waypoints=gt_wp,
            pred_meta=out["meta_logits"],
            gt_meta=meta,
            waypoint_mask=wp_mask,
        )
        total_loss += losses["total"].item()

        # ADE (Average Displacement Error)
        pred_xy = out["waypoints"][..., :2]
        gt_xy = gt_wp[..., :2]
        displacement = (pred_xy - gt_xy).norm(dim=-1)  # (B, T)
        ade = (displacement * wp_mask).sum() / wp_mask.sum().clamp(min=1)
        ade_sum += ade.item()

        # FDE (Final Displacement Error) — error at t=4.0s
        B = pred_xy.shape[0]
        # Find last valid index per sample
        last_valid = wp_mask.long().cumsum(dim=1).max(dim=1).values - 1  # (B,)
        last_valid = last_valid.clamp(min=0)
        fde_batch = 0.0
        for b in range(B):
            t = last_valid[b].item()
            fde_batch += (pred_xy[b, t] - gt_xy[b, t]).norm().item()
        fde_sum += fde_batch / B

        n_batches += 1

        avg_ade = ade_sum / n_batches
        avg_fde = fde_sum / n_batches
        pbar.set_postfix(ADE=f"{avg_ade:.2f}m", FDE=f"{avg_fde:.2f}m")

    pbar.close()

    return (
        total_loss / max(n_batches, 1),
        ade_sum / max(n_batches, 1),
        fde_sum / max(n_batches, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="HALO Planner Training")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to nuScenes dataset")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print header
    print("=" * 65)
    print("  HALO Planner — Training")
    print("=" * 65)
    print(f"  Device:       {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  Dataset:      {args.version}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Model dim:    {args.d_model}")
    print(f"  Enc layers:   {args.encoder_layers}")
    print(f"  Dec layers:   {args.decoder_layers}")
    print("=" * 65)

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb_logs"))

    # --- Dataset ---
    split_map = {
        "v1.0-mini": ("mini_train", "mini_val"),
        "v1.0-trainval": ("train", "val"),
    }
    train_split, val_split = split_map.get(args.version, ("train", "val"))

    print("\nLoading training set...")
    train_ds = NuScenesDataset(args.dataroot, args.version, split=train_split)
    print("Loading validation set...")
    val_ds = NuScenesDataset(args.dataroot, args.version, split=val_split)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print(f"\n  Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds):,} samples ({len(val_loader)} batches)")

    # --- Model ---
    model = HaloPlanner(
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
    ).to(device)

    param_count = model.count_parameters()
    print(f"  Model: {param_count:,} parameters ({param_count / 1e6:.1f}M)")

    # --- Optimiser ---
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    criterion = PlannerLoss()

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val_ade = float("inf")
    global_step = 0

    if args.resume and os.path.isfile(args.resume):
        print(f"\n  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_ade = ckpt.get("val_ade", float("inf"))
        global_step = ckpt.get("global_step", 0)
        # Advance scheduler to correct position
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, best ADE: {best_val_ade:.3f}m")

    # --- Training loop ---
    history = []
    epoch_times = []
    training_start = time.time()

    print(f"\n{'='*65}")
    print(f"  Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_components, train_ade, global_step = train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch, writer, global_step
        )

        # Validate
        val_loss, val_ade, val_fde = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        lr_now = scheduler.get_last_lr()[0]

        # ETA calculation
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        total_elapsed = time.time() - training_start

        # Epoch summary
        is_best = val_ade < best_val_ade
        best_marker = " *BEST*" if is_best else ""

        print(f"\n  Epoch {epoch}/{args.epochs} summary:")
        print(f"    Train loss: {train_loss:.4f}  |  Train ADE: {train_ade:.3f}m")
        print(f"    Val loss:   {val_loss:.4f}  |  Val ADE:   {val_ade:.3f}m  |  Val FDE: {val_fde:.3f}m{best_marker}")
        print(f"    Components: traj={train_components['traj']:.4f}  hdg={train_components['heading']:.4f}  "
              f"vel={train_components['velocity']:.4f}  meta={train_components['meta']:.4f}  "
              f"coll={train_components['collision']:.4f}")
        print(f"    LR: {lr_now:.2e}  |  Time: {format_time(epoch_time)}  |  "
              f"Elapsed: {format_time(total_elapsed)}  |  ETA: {format_time(eta_seconds)}")

        # TensorBoard epoch logging
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/train_ADE", train_ade, epoch)
        writer.add_scalar("epoch/val_ADE", val_ade, epoch)
        writer.add_scalar("epoch/val_FDE", val_fde, epoch)
        writer.add_scalar("epoch/lr", lr_now, epoch)
        for k, v in train_components.items():
            writer.add_scalar(f"epoch/component_{k}", v, epoch)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ade": train_ade,
            "val_loss": val_loss,
            "val_ade": val_ade,
            "val_fde": val_fde,
            "lr": lr_now,
            "epoch_time": epoch_time,
            **{f"train_{k}": v for k, v in train_components.items()},
        })

        # Save best model
        if is_best:
            best_val_ade = val_ade
            path = os.path.join(args.save_dir, "best_planner.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_ade": val_ade,
                "val_fde": val_fde,
                "global_step": global_step,
                "args": vars(args),
            }, path)
            print(f"    Saved best model (ADE={val_ade:.3f}m, FDE={val_fde:.3f}m)")

        # Save latest (always)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "val_ade": val_ade,
            "val_fde": val_fde,
            "global_step": global_step,
            "args": vars(args),
        }, os.path.join(args.save_dir, "latest_planner.pt"))

        # Save history every epoch (so you can inspect mid-training)
        with open(os.path.join(args.save_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print()

    writer.close()

    # Final summary
    total_time = time.time() - training_start
    print("=" * 65)
    print(f"  Training complete")
    print(f"  Total time:    {format_time(total_time)}")
    print(f"  Best val ADE:  {best_val_ade:.3f}m")
    print(f"  Checkpoints:   {args.save_dir}/")
    print(f"  TensorBoard:   tensorboard --logdir {args.save_dir}/tb_logs")
    print("=" * 65)


if __name__ == "__main__":
    main()
