import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path so we can import the dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halo_planner.dataset_nuscenes import NuScenesDataset
from halo_planner.model import MAX_LANE_POINTS

def plot_scene(data_dict, title="HALO Planner Scene"):
    """Visualises a single extracted scene dictionary."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract numpy arrays from tensors
    lanes = data_dict["lanes"].numpy()
    lane_mask = data_dict["lane_mask"].numpy()
    objects = data_dict["objects"].numpy()
    obj_mask = data_dict["object_mask"].numpy()
    waypoints = data_dict["gt_waypoints"].numpy()
    wp_mask = data_dict["waypoint_mask"].numpy()
    
    # 1. Plot Lanes (Grey)
    for i in range(len(lane_mask)):
        if lane_mask[i]:
            # Extract the x,y coordinates (first MAX_LANE_POINTS * 2 elements)
            pts_flat = lanes[i, :MAX_LANE_POINTS * 2]
            pts = pts_flat.reshape(MAX_LANE_POINTS, 2)
            ax.plot(pts[:, 0], pts[:, 1], color='lightgray', linewidth=2, linestyle='--')

    # 2. Plot Objects (Red boxes + Blue velocity vectors)
    for i in range(len(obj_mask)):
        if obj_mask[i]:
            x, y = objects[i, 0], objects[i, 1]
            w, l = objects[i, 3], objects[i, 4]
            yaw = objects[i, 6]
            vx, vy = objects[i, 14], objects[i, 15]
            
            # Create a bounding box
            rect = patches.Rectangle(
                (x - l/2, y - w/2), l, w, angle=np.degrees(yaw), 
                rotation_point='center', linewidth=1.5, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Plot velocity vector (where the object will be in 1 second)
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                ax.arrow(x, y, vx, vy, head_width=0.5, head_length=0.5, fc='blue', ec='blue')

    # 3. Plot Ground Truth Waypoints (Green dots)
    valid_wps = waypoints[wp_mask]
    if len(valid_wps) > 0:
        ax.plot(valid_wps[:, 0], valid_wps[:, 1], marker='o', color='green', markersize=4, label='GT Path')
        
    # 4. Plot Ego Vehicle (Blue dot at Origin)
    ax.plot(0, 0, marker='s', color='blue', markersize=10, label='Ego (Origin)')

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(-20, 60)  # 20m behind, 60m ahead
    ax.set_ylim(-30, 30)  # 30m left/right
    ax.set_xlabel("X (forward) [m]")
    ax.set_ylabel("Y (left) [m]")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    # Make sure you point this to your actual nuScenes dataroot
    dataroot = "./data/nuscenes"  # Change this if your path is different!
    
    print("Loading dataset (this takes a moment)...")
    # Using 'train' split so the 50% trajectory perturbation triggers
    dataset = NuScenesDataset(dataroot=dataroot, version="v1.0-trainval", split="train")
    
    print("\nVisualising 3 random samples...")
    # Pick a few random indices
    indices = np.random.choice(len(dataset), 3, replace=False)
    
    for idx in indices:
        data = dataset[idx]
        
        # Check if this frame got perturbed by looking at the first waypoint.
        # If the first waypoint is NOT (0,0), it means the base frame was shifted!
        y_start = data["gt_waypoints"][0, 1].item()
        if abs(y_start) > 0.1:
            title = f"Sample {idx} — PERTURBED (Recovery Path)"
        else:
            title = f"Sample {idx} — Normal Centered Frame"
            
        print(f"Plotting {title}")
        plot_scene(data, title=title)

if __name__ == "__main__":
    main()
