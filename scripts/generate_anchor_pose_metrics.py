#!/usr/bin/env python3
"""
Generate pose_metrics.xlsx from existing anchor output (chosen initial_pose + gt_pose + GT mesh).
Use when anchor was already run and mesh/pose exist but pose_metrics.xlsx is missing.
Usage:
  python scripts/generate_anchor_pose_metrics.py \
    --anchor_folder /path/to/anchor_results/ycbvineoat/dexycb_reference_view_ours \
    --ycb_model_path /path/to/YCB_Video_Models
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import trimesh

# Allow importing from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import compute_add, compute_adds, compute_RT_distances

OBJ_NUM_MAP = {
    "003_cracker_box": 2,
    "006_mustard_bottle": 5,
    "021_bleach_cleanser": 12,
    "019_pitcher_base": 11,
    "004_sugar_box": 3,
    "005_tomato_soup_can": 4,
    "010_potted_meat_can": 9,
}


def main():
    parser = argparse.ArgumentParser(description="Generate pose_metrics.xlsx from existing anchor outputs")
    parser.add_argument("--anchor_folder", type=str, required=True, help="Path to anchor results folder")
    parser.add_argument("--ycb_model_path", type=str, required=True, help="Path to YCB Video Models (models/obj/)")
    args = parser.parse_args()

    anchor_folder = os.path.abspath(args.anchor_folder)
    ycb_model_path = os.path.abspath(args.ycb_model_path)
    if not os.path.isdir(anchor_folder):
        print(f"Anchor folder not found: {anchor_folder}")
        sys.exit(1)

    obj_list = [
        d for d in os.listdir(anchor_folder)
        if os.path.isdir(os.path.join(anchor_folder, d))
        and not d.startswith(".")
    ]
    pose_rows = []
    for obj in sorted(obj_list):
        obj_dir = os.path.join(anchor_folder, obj)
        initial_pose_path = os.path.join(obj_dir, f"{obj}_initial_pose.txt")
        gt_pose_path = os.path.join(obj_dir, f"{obj}_gt_pose.txt")
        gt_mesh_path = os.path.join(ycb_model_path, "models", obj, "textured_simple.obj")
        if not os.path.isfile(initial_pose_path):
            print(f"Skip {obj}: missing {obj}_initial_pose.txt")
            continue
        if not os.path.isfile(gt_pose_path):
            print(f"Skip {obj}: missing {obj}_gt_pose.txt")
            continue
        if not os.path.isfile(gt_mesh_path):
            print(f"Skip {obj}: missing GT mesh {gt_mesh_path}")
            continue

        pred_pose = np.loadtxt(initial_pose_path)
        gt_pose = np.loadtxt(gt_pose_path)
        gt_mesh = trimesh.load(gt_mesh_path)

        add_val = float(compute_add(gt_mesh.vertices, pred_pose, gt_pose))
        adds_val = float(compute_adds(gt_mesh.vertices, pred_pose, gt_pose))
        err_R, err_T = compute_RT_distances(pred_pose, gt_pose)
        err_R = float(np.asarray(err_R).reshape(-1)[0])
        err_T = float(np.asarray(err_T).reshape(-1)[0])

        obj_num = OBJ_NUM_MAP.get(obj, -1)
        pose_rows.append({
            "Object": obj,
            "Object_Number": obj_num,
            "Chosen_Mesh": "from_file",
            "ADD_S": adds_val,
            "ADD": add_val,
            "R_error": err_R,
            "T_error": err_T,
        })
        print(f"{obj}: ADD-S={adds_val:.6f} ADD={add_val:.6f} R_error={err_R:.3f} T_error={err_T:.3f}")

    if not pose_rows:
        print("No objects processed. Exiting.")
        sys.exit(1)

    pose_df = pd.DataFrame(pose_rows).sort_values("Object")
    pose_excel_path = os.path.join(anchor_folder, "pose_metrics.xlsx")
    pose_df.to_excel(pose_excel_path, index=False, sheet_name="chosen")
    print(f"\nPose metrics saved to: {pose_excel_path}")


if __name__ == "__main__":
    main()
