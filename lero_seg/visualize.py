"""Pictures of per-point lidar classes: bird's-eye view, range-image strip, legend, coloured .ply."""
from pathlib import Path

import cv2
import numpy as np

BACKGROUND = 30  # grey; black is the ignore colour


def bev(points: np.ndarray, point_classes: np.ndarray, palette: np.ndarray, size=500, metres=50.0) -> np.ndarray:
    """Top view (x up, y left) of `metres` x `metres` around the sensor, points as 2x2 px dots."""
    img = np.full((size, size, 3), BACKGROUND, np.uint8)
    scale = size / metres
    u = (size / 2 - points[:, 1] * scale).astype(np.int64)
    v = (size / 2 - points[:, 0] * scale).astype(np.int64)
    ok = (u >= 0) & (u < size - 1) & (v >= 0) & (v < size - 1)
    for du in (0, 1):
        for dv in (0, 1):
            img[v[ok] + dv, u[ok] + du] = palette[point_classes[ok]]
    return img


def range_strip(class_img: np.ndarray, mask: np.ndarray, palette: np.ndarray, row_px=6) -> np.ndarray:
    """The (H, W) range image's classes, each laser layer `row_px` px tall; empty pixels grey."""
    img = np.where(mask[..., None], palette[class_img], BACKGROUND).astype(np.uint8)
    return np.repeat(img, row_px, axis=0)


def legend(names, palette: np.ndarray, height: int, width=130) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    for i, name in enumerate(names):
        y = 10 + 22 * i
        cv2.rectangle(img, (6, y), (20, y + 14), palette[i].tolist(), -1)
        cv2.putText(img, name, (26, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def title(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    """Binary PLY with xyz + rgb, e.g. for CloudCompare or MeshLab."""
    vertex = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'], vertex['y'], vertex['z'] = points[:, 0], points[:, 1], points[:, 2]
    vertex['red'], vertex['green'], vertex['blue'] = colours[:, 0], colours[:, 1], colours[:, 2]
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(points)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(vertex.tobytes())
