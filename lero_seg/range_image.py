"""Spherical projection of one SICK multiScan165 scan onto a (layers x width) range image."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .recordings import read_jsonl, recorder_classes

CHANNELS = ('range', 'x', 'y', 'z', 'intensity')


@dataclass
class Projection:
    image: np.ndarray   # (5, H, W) float32: range, x, y, z, log10 intensity; 0 where empty
    mask: np.ndarray    # (H, W) bool: pixel holds a point
    row: np.ndarray     # (N,) per-point row, -1 if the point matched no layer
    col: np.ndarray     # (N,) per-point column
    index: np.ndarray   # (H, W) index of the point kept in each pixel (nearest), -1 if empty


def project(points: np.ndarray, layers_deg, width: int, tolerance_deg: float = 0.6) -> Projection:
    xyz = points[:, :3].astype(np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    elevation = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(rng, 1e-6), -1, 1)))
    layers = np.asarray(layers_deg, dtype=np.float64)
    diff = np.abs(elevation[:, None] - layers[None, :])
    row = diff.argmin(axis=1)
    ok = (diff.min(axis=1) <= tolerance_deg) & (rng > 0)
    row = np.where(ok, row, -1)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    col = (np.floor((azimuth + np.pi) / (2 * np.pi) * width).astype(np.int64)) % width

    height = len(layers)
    index = np.full((height, width), -1, dtype=np.int64)
    order = np.flatnonzero(ok)[np.argsort(-rng[ok])]  # far first, so the nearest point wins a pixel
    index[row[order], col[order]] = order

    mask = index >= 0
    kept = index[mask]
    image = np.zeros((len(CHANNELS), height, width), dtype=np.float32)
    intensity = np.log10(np.maximum(points[:, 3], 0) + 1e-9) if points.shape[1] > 3 else np.zeros(len(points))
    for c, values in enumerate((rng, xyz[:, 0], xyz[:, 1], xyz[:, 2], intensity)):
        image[c][mask] = values[kept]
    return Projection(image, mask, row, col, index)


def lidar_frames(recording: Path) -> list[dict]:
    """The recording's scans that have per-point labels: [{frame, topic, frame_id, stamp_ns, robot_pose,
    points, labels}]."""
    out = []
    for entry in read_jsonl(recording / 'lidar_index.jsonl'):
        points = recording / 'points' / f'{entry["frame"]}.npy'
        labels = recording / 'point_labels' / f'{entry["frame"]}.npy'
        if points.is_file() and labels.is_file():
            out.append({'recording': recording, 'frame': entry['frame'], 'topic': entry.get('topic'),
                        'frame_id': entry.get('frame_id'), 'stamp_ns': entry.get('stamp_ns'),
                        'robot_pose': entry.get('robot_pose'), 'points': points, 'labels': labels})
    return out


def load_scan(item: dict, classes) -> tuple[np.ndarray, np.ndarray]:
    """Points (N, 4) and their train ids (N,)."""
    lut = classes.lidar_lut(recorder_classes(item['recording']))
    return np.load(item['points']), lut[np.load(item['labels']).reshape(-1).astype(np.int64)]
