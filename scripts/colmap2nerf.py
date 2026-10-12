#!/usr/bin/env python3
"""
colmap2nerf.py

Converts COLMAP sparse reconstruction (sparse/0) into NeRF/Skyfall-GS format:
- transforms_train.json
- transforms_test.json
- points3D.txt
- points3D.ply
"""

import os
import sys
import json
import argparse
import numpy as np

# Add parent directory to sys.path so we can import from scene
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary,
    read_extrinsics_text, read_intrinsics_text, read_points3D_text,
    qvec2rotmat
)
from scene.dataset_readers import storePly

def convert(sparse_dir, output_dir, images_folder="images", test_holdout=8):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Read cameras & extrinsics
    try:
        extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
        
    # 2. Read point cloud
    ply_path = os.path.join(output_dir, "points3D.ply")
    txt_path = os.path.join(output_dir, "points3D.txt")
    
    xyz, rgb = None, None
    try:
        bin_pts = os.path.join(sparse_dir, "points3D.bin")
        if os.path.isfile(bin_pts):
            pts_dict = read_points3D_binary(bin_pts)
            xyz = np.array([p.xyz for p in pts_dict.values()])
            rgb = np.array([p.rgb for p in pts_dict.values()])
            # Export points3D.txt for loaders requiring txt
            with open(txt_path, "w") as f:
                f.write(f"# Number of points: {len(pts_dict)}\n")
                for pid, p in pts_dict.items():
                    tracks_str = " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in zip(p.image_ids, p.point2D_idxs))
                    f.write(f"{pid} {p.xyz[0]} {p.xyz[1]} {p.xyz[2]} {p.rgb[0]} {p.rgb[1]} {p.rgb[2]} {p.error} {tracks_str}\n")
    except Exception as e:
        print(f"[colmap2nerf] Binary point cloud notice: {e}")
        
    if xyz is None:
        try:
            txt_pts = os.path.join(sparse_dir, "points3D.txt")
            if os.path.isfile(txt_pts):
                xyz, rgb, _ = read_points3D_text(txt_pts)
                if not os.path.isfile(txt_path):
                    import shutil
                    shutil.copy2(txt_pts, txt_path)
        except Exception as e:
            print(f"[colmap2nerf] Text point cloud notice: {e}")
            
    if xyz is not None and rgb is not None:
        storePly(ply_path, xyz, rgb)
        print(f"[colmap2nerf] Saved point cloud with {len(xyz)} points to {ply_path}")
        
    # 3. Build frames for transforms_train.json & transforms_test.json
    frames_train = []
    frames_test = []
    
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)
    for idx, img in enumerate(sorted_images):
        cam = intrinsics[img.camera_id]
        
        # Determine focal length and principal points based on camera model
        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fl_x = float(cam.params[0])
            fl_y = float(cam.params[0])
            cx = float(cam.params[1])
            cy = float(cam.params[2])
        elif cam.model in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE"):
            fl_x = float(cam.params[0])
            fl_y = float(cam.params[1])
            cx = float(cam.params[2])
            cy = float(cam.params[3])
        else:
            fl_x = float(cam.params[0])
            fl_y = float(cam.params[0])
            cx = float(cam.width) / 2.0
            cy = float(cam.height) / 2.0
            
        R = qvec2rotmat(img.qvec)
        T = np.array(img.tvec)
        
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        
        c2w = np.linalg.inv(w2c)
        
        frame = {
            "file_path": f"./{images_folder}/{img.name}",
            "transform_matrix": c2w.tolist(),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": int(cam.width),
            "h": int(cam.height)
        }
        
        if test_holdout > 0 and idx % test_holdout == 0:
            frames_test.append(frame)
        else:
            frames_train.append(frame)
            
    if len(frames_test) == 0:
        frames_test = frames_train[:1]
        
    train_data = {
        "camera_model": "PINHOLE",
        "frames": frames_train
    }
    test_data = {
        "camera_model": "PINHOLE",
        "frames": frames_test
    }
    
    train_json_path = os.path.join(output_dir, "transforms_train.json")
    test_json_path = os.path.join(output_dir, "transforms_test.json")
    
    with open(train_json_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_json_path, "w") as f:
        json.dump(test_data, f, indent=2)
        
    print(f"[colmap2nerf] Successfully generated {train_json_path} ({len(frames_train)} frames) and {test_json_path} ({len(frames_test)} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COLMAP to NeRF/Skyfall transforms")
    parser.add_argument("--sparse_dir", required=True, help="Path to COLMAP sparse/0 directory")
    parser.add_argument("--output_dir", required=True, help="Output scene directory")
    parser.add_argument("--images_folder", default="images", help="Relative path to images folder")
    parser.add_argument("--holdout", type=int, default=8, help="Holdout interval for test split")
    args = parser.parse_args()
    
    convert(args.sparse_dir, args.output_dir, args.images_folder, args.holdout)
