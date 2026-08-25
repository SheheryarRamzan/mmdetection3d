"""Convert PandaSet .pkl point clouds to .bin and .pcd formats.

Converts world-frame point clouds to ego-centric frame and saves as:
  - .bin: raw float32 binary (x, y, z, intensity) per point
  - .pcd: ASCII PCD format

Output structure:
  /swmotion-user-data/datasets/pandaset/cloud_bin/pandar64/{seq}/{frame}.bin
  /swmotion-user-data/datasets/pandaset/cloud_bin/pandargt/{seq}/{frame}.bin
  /swmotion-user-data/datasets/pandaset/cloud_pcd/pandar64/{seq}/{frame}.pcd
  /swmotion-user-data/datasets/pandaset/cloud_pcd/pandargt/{seq}/{frame}.pcd
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from functools import partial


PANDASET_ROOT = '/swmotion-user-data/datasets/pandaset/sequences'
OUT_BIN = '/swmotion-user-data/datasets/pandaset/cloud_bin'
OUT_PCD = '/swmotion-user-data/datasets/pandaset/cloud_pcd'

SKIP_DIRS = {'pkls', 'cloud_bin', 'cloud_pcd'}


def pose_to_matrix(pose):
    """Convert pose dict (position + heading quaternion) to 4x4 matrix."""
    from scipy.spatial.transform import Rotation as R
    q = pose['heading']
    rot = R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    pos = pose['position']
    mat[:3, 3] = [pos['x'], pos['y'], pos['z']]
    return mat


def write_pcd(filepath, points):
    """Write points (N, 4) as ASCII PCD file."""
    n = len(points)
    with open(filepath, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for i in range(n):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} {points[i,3]:.6f}\n")


def write_pcd_binary(filepath, points):
    """Write points (N, 4) as binary PCD file (much faster than ASCII)."""
    n = len(points)
    points_f32 = points.astype(np.float32)
    with open(filepath, 'wb') as f:
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z intensity\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {n}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n}\n"
            "DATA binary\n"
        )
        f.write(header.encode('ascii'))
        f.write(points_f32.tobytes())


def process_sequence(seq_id):
    """Process all frames in a sequence."""
    seq_dir = os.path.join(PANDASET_ROOT, seq_id)
    lidar_dir = os.path.join(seq_dir, 'lidar')
    poses_file = os.path.join(lidar_dir, 'poses.json')

    if not os.path.isfile(poses_file):
        return f'{seq_id}: SKIP (no poses.json)'

    with open(poses_file, 'r') as f:
        poses = json.load(f)

    frame_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.pkl')])
    converted = 0

    for frame_file in frame_files:
        frame_idx = int(frame_file.replace('.pkl', ''))
        frame_path = os.path.join(lidar_dir, frame_file)

        # Load point cloud
        with open(frame_path, 'rb') as f:
            df = pickle.load(f)

        # Get pose and build inverse transform
        pose = poses[frame_idx]
        pose_mat = pose_to_matrix(pose)
        inv_pose = np.linalg.inv(pose_mat)

        # Extract columns
        world_xyz = df[['x', 'y', 'z']].values.astype(np.float64)
        intensity = df['i'].values.astype(np.float32)
        device = df['d'].values  # 0=Pandar64, 1=PandarGT

        # Transform to ego frame
        pts_h = np.hstack([world_xyz, np.ones((len(world_xyz), 1))])
        ego_xyz = (inv_pose @ pts_h.T).T[:, :3].astype(np.float32)

        # Split by sensor
        mask_p64 = (device == 0)
        mask_pgt = (device == 1)

        frame_name = f'{frame_idx:02d}'

        for sensor_name, mask in [('pandar64', mask_p64), ('pandargt', mask_pgt)]:
            if mask.sum() == 0:
                continue

            pts = np.column_stack([ego_xyz[mask], intensity[mask]]).astype(np.float32)

            # .bin output
            bin_dir = os.path.join(OUT_BIN, sensor_name, seq_id)
            os.makedirs(bin_dir, exist_ok=True)
            bin_path = os.path.join(bin_dir, f'{frame_name}.bin')
            pts.tofile(bin_path)

            # .pcd output
            pcd_dir = os.path.join(OUT_PCD, sensor_name, seq_id)
            os.makedirs(pcd_dir, exist_ok=True)
            pcd_path = os.path.join(pcd_dir, f'{frame_name}.pcd')
            write_pcd_binary(pcd_path, pts)

        converted += 1

    return f'{seq_id}: {converted} frames converted'


def main():
    # Get all sequence directories
    all_entries = sorted(os.listdir(PANDASET_ROOT))
    sequences = [e for e in all_entries if e not in SKIP_DIRS and
                 os.path.isdir(os.path.join(PANDASET_ROOT, e, 'lidar'))]

    print(f'Found {len(sequences)} sequences')
    print(f'Output .bin: {OUT_BIN}')
    print(f'Output .pcd: {OUT_PCD}')
    print()

    # Process with multiprocessing
    with Pool(processes=8) as pool:
        results = pool.map(process_sequence, sequences)

    for r in results:
        print(r)

    # Summary
    total_bin = 0
    for sensor in ['pandar64', 'pandargt']:
        sensor_dir = os.path.join(OUT_BIN, sensor)
        if os.path.exists(sensor_dir):
            for seq in os.listdir(sensor_dir):
                seq_path = os.path.join(sensor_dir, seq)
                if os.path.isdir(seq_path):
                    total_bin += len(os.listdir(seq_path))

    print(f'\nTotal .bin files: {total_bin}')
    print('Done!')


if __name__ == '__main__':
    main()
