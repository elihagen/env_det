"""Rigid transforms from a recording's URDF and robot poses, and projecting lidar points into a camera.

A scan and a camera frame are recorded at slightly different times while the robot drives, so
points go lidar -> base_link (static, URDF) -> world (robot pose at the scan) -> base_link (robot
pose at the camera frame) -> camera (static, URDF) -> image.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# The recorder's camera mounts are USD cameras (look along -Z, +Y up); OpenCV looks along +Z, +Y down.
USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])


def quat_matrix(x, y, z, w) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_matrix(roll, pitch, yaw) -> np.ndarray:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(rotation: np.ndarray, translation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, translation
    return out


def pose_matrix(pose) -> np.ndarray:
    """world <- base_link from a recorded robot_pose [x, y, z, qx, qy, qz, qw]."""
    return transform(quat_matrix(*pose[3:7]), pose[:3])


def urdf_transform(urdf: Path, parent: str, child: str) -> np.ndarray:
    """parent <- child through the URDF's (fixed) joints; `parent` must be an ancestor of `child`."""
    joints = {}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0, 0, 0]
        joints[joint.find('child').get('link')] = (joint.find('parent').get('link'), transform(rpy_matrix(*rpy), xyz))
    out, link = np.eye(4), child
    while link != parent:
        if link not in joints:
            raise KeyError(f'{child} is not below {parent} in {urdf}')
        link, step = joints[link]
        out = step @ out
    return out


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]


def camera_intrinsics(cam_dir: Path) -> dict | None:
    path = cam_dir / 'camera_info.json'
    if not path.is_file():
        return None
    with open(path) as f:
        info = json.load(f)
    return {'K': np.asarray(info['K'], dtype=np.float64).reshape(3, 3), 'width': int(info['width']),
            'height': int(info['height'])}


def project_to_image(points_cam: np.ndarray, K: np.ndarray, width: int, height: int, min_depth=0.1,
                     occlusion_cell=8, occlusion_tol=(0.05, 0.3)):
    """Pixel (u, v) of each point in the optical frame, and whether the camera sees it.

    A point is hidden if a nearer point falls into the same `occlusion_cell` px cell by more than
    rel * depth + abs metres (`occlusion_tol`): the lidar sees things the offset camera can't.
    """
    depth = points_cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = points_cam[:, :2] / depth[:, None] @ K[:2, :2].T + K[:2, 2]
    valid = (depth > min_depth) & np.isfinite(uv).all(1)
    uv = np.where(valid[:, None], uv, -1).astype(np.int64)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if occlusion_cell and valid.any():
        cells_w = -(-width // occlusion_cell)
        cell = (uv[:, 1] // occlusion_cell) * cells_w + uv[:, 0] // occlusion_cell
        nearest = np.full(cells_w * -(-height // occlusion_cell), np.inf)
        np.minimum.at(nearest, cell[valid], depth[valid])
        rel, abs_ = occlusion_tol
        valid &= depth <= nearest[np.where(valid, cell, 0)] * (1 + rel) + abs_
    return uv, valid
