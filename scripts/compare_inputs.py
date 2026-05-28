"""
Side-by-side input comparison: nuScenes (in-distribution, model works) vs
MetaDrive (out-of-distribution, model drifts). Finds the field where the two
input distributions diverge — that field is the bug.

This is the disciplined version of the debugging we've been doing by eye. It
dumps statistics per input tensor field for N samples from each source so a
scale/sign/range mismatch is visible as a number, not a guess.

Run:
    python scripts/compare_inputs.py \
        --checkpoint checkpoints/best_planner.pt \
        --dataroot ~/halo_planner/data/nuscenes --version v1.0-trainval
"""
import argparse, numpy as np, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from halo_planner.model import HaloPlanner, MAX_LANE_POINTS
from halo_planner.dataset_nuscenes import NuScenesDataset


def stats(name, arr):
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        print(f"    {name:16s}: empty")
        return
    print(f"    {name:16s}: min={a.min():+8.2f} max={a.max():+8.2f} "
          f"mean={a.mean():+8.2f} std={a.std():7.2f} nonzero={np.count_nonzero(a)}/{a.size}")


def dump_scene(tag, ego, objs, objm, lanes, lanem, nav):
    print(f"\n  === {tag} ===")
    print(f"    n_objects={int(objm.sum())}  n_lanes={int(lanem.sum())}  nav={np.round(nav,2)}")
    stats("ego_state", ego)
    if objm.sum() > 0:
        ov = objs[objm.astype(bool)]
        stats("obj x", ov[:, 0]); stats("obj y", ov[:, 1])
        stats("obj w,l,h", ov[:, 3:6]); stats("obj yaw", ov[:, 6])
        stats("obj velX,velY", ov[:, 14:16])
    if lanem.sum() > 0:
        lv = lanes[lanem.astype(bool)]
        xy = lv[:, :MAX_LANE_POINTS*2].reshape(len(lv), MAX_LANE_POINTS, 2)
        stats("lane x", xy[..., 0]); stats("lane y", xy[..., 1])
        # per-lane forward extent
        ext = xy[:, :, 0].max(axis=1) - xy[:, :, 0].min(axis=1)
        stats("lane x-extent", ext)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--map", default="SSS")
    args = ap.parse_args()

    # --- nuScenes sample (in-distribution) ---
    ds = NuScenesDataset(args.dataroot, version=args.version, split="val")
    # pick a MOVING straight sample (meta=follow) for fair comparison
    s = None
    for i in np.linspace(0, len(ds)-1, 40).astype(int):
        cand = ds[i]
        if int(cand["meta_action"]) == 0:  # follow_lane
            s = cand; break
    if s is None:
        s = ds[0]
    dump_scene("nuScenes (model WORKS here)",
               s["ego_state"].numpy(), s["objects"].numpy(), s["object_mask"].numpy(),
               s["lanes"].numpy(), s["lane_mask"].numpy(), s["nav_command"].numpy())

    # --- MetaDrive sample (out-of-distribution) ---
    from metadrive.envs.metadrive_env import MetaDriveEnv
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_metadrive import scene_to_tensors
    env = MetaDriveEnv(dict(use_render=False, num_scenarios=1, map=args.map, traffic_density=0.0))
    env.reset()
    for _ in range(10):  # let it settle / start moving a touch
        env.step([0, 0.3])
    md = scene_to_tensors(env, torch.device("cpu"))
    dump_scene("MetaDrive (model DRIFTS here)",
               md["ego_state"][0].numpy(), md["objects"][0].numpy(), md["object_mask"][0].numpy(),
               md["lanes"][0].numpy(), md["lane_mask"][0].numpy(), md["nav_command"][0].numpy())
    env.close()

    print("\n  READ: compare the two blocks field by field. The field whose")
    print("  distribution differs most (scale, sign, range) is the adapter bug.")
    print("  Prime suspects: lane x/y range & extent, ego_state scale, obj dims.")


if __name__ == "__main__":
    main()
