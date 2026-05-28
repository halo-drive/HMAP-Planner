"""
Open-loop nuScenes verification — the missing test.

Feeds REAL nuScenes training/val samples through the model and compares the
predicted waypoints to the LOGGED ground-truth future trajectory. This isolates
the planner from MetaDrive and the controller entirely.

Run from repo root:
    python scripts/openloop_nuscenes_check.py \
        --checkpoint checkpoints/best_planner.pt \
        --dataroot /path/to/nuscenes --version v1.0-mini --n 10

What to read:
  * If predicted ~ ground-truth (low per-sample ADE, straight when GT straight)
    -> the MODEL IS SOUND. The whole problem is the MetaDrive adapter mismatch.
    You then fix scene_to_tensors/extract_lanes to match dataset_nuscenes.py
    field-for-field (NOT to be "cleaner").
  * If predicted diverges from GT even on its own training distribution
    -> the problem is upstream (loss / labels / ADE masking). Retraining needed.
"""
import argparse, numpy as np, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from halo_planner.model import HaloPlanner, META_ACTIONS
from halo_planner.dataset_nuscenes import NuScenesDataset


def ade(pred, gt, mask):
    d = np.linalg.norm(pred[mask, :2] - gt[mask, :2], axis=1)
    return float(d.mean()) if d.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HaloPlanner()
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.to(dev).eval()
    print(f"Loaded epoch {ck['epoch']} val_ade={ck.get('val_ade','?')}")

    ds = NuScenesDataset(args.dataroot, version=args.version, split=args.split)
    n = min(args.n, len(ds))
    print(f"Checking {n} samples\n")

    ades = []
    for i in range(n):
        s = ds[i]
        inp = {k: s[k].unsqueeze(0).to(dev)
               for k in ["ego_state","objects","object_mask","lanes","lane_mask","nav_command"]}
        with torch.no_grad():
            out = model(**inp)
        pred = out["waypoints"][0].cpu().numpy()
        meta_pred = int(out["meta_logits"][0].argmax())
        gt = s["gt_waypoints"].numpy()
        m = s["waypoint_mask"].numpy().astype(bool)
        a = ade(pred, gt, m)
        ades.append(a)

        # The key comparison: where does GT go vs where does pred go (at horizon)?
        last = np.where(m)[0].max() if m.any() else 0
        print(f"sample {i:2d}: ADE={a:5.2f}m | "
              f"GT_end=({gt[last,0]:+5.1f},{gt[last,1]:+5.1f}) "
              f"PRED_end=({pred[last,0]:+5.1f},{pred[last,1]:+5.1f}) | "
              f"meta gt={META_ACTIONS[int(s['meta_action'])]:<12} pred={META_ACTIONS[meta_pred]}")

    ades = np.array(ades)
    print(f"\nMean open-loop ADE: {ades.mean():.2f}m (ckpt reported {ck.get('val_ade','?')})")
    print("READ: if this ~matches the reported val_ade and PRED_end tracks GT_end,")
    print("      the model is sound -> fix the MetaDrive adapter to match dataset_nuscenes.py.")
    print("      if PRED_end ignores GT_end (esp. lateral sign), the model didn't learn the task.")


if __name__ == "__main__":
    main()