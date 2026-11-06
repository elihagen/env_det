"""Per-scan environment detection on the robot: SalsaNext on one lidar scan, optionally fused with
RF-DETR-Seg on the camera frames nearest in time (see fusion.py), plus per-scan object ids.

ROS-free, so it runs on recordings as well (scripts/env_det_node.py wraps it for ROS 2). The scan is
used exactly as the recorder stored it for training: (N, 3|4) x, y, z[, intensity] in the lidar frame.
"""
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .classes import Classes
from .fusion import camera_point_probs, fuse, learned_fuse, lidar_point_probs, load_head
from .geometry import apply, project_to_image
from .lidar_dataset import normalize
from .range_image import project
from .salsanext import SalsaNext

RFDETR_SIZES = {'nano': 'RFDETRSegNano', 'small': 'RFDETRSegSmall', 'medium': 'RFDETRSegMedium',
                'large': 'RFDETRSegLarge', 'xlarge': 'RFDETRSegXLarge', '2xlarge': 'RFDETRSeg2XLarge'}


@dataclass
class CameraView:
    """One camera frame to fuse with a scan. `optical_from_lidar` maps scan points into the camera's
    OpenCV optical frame (+Z forward, +Y down) at the frame's time; `key` identifies the frame, so its
    detections are computed once even when several scans use it."""
    key: tuple
    rgb: np.ndarray
    K: np.ndarray
    width: int
    height: int
    optical_from_lidar: np.ndarray


@dataclass
class ScanResult:
    labels: np.ndarray       # (N,) uint8 train id, classes.ignore where no sensor saw the point
    confidence: np.ndarray   # (N,) float32 fused probability of that class, 0 where unseen
    cameras_used: list = field(default_factory=list)


class EnvDetector:
    def __init__(self, salsanext: str, device: str = 'cuda', rfdetr: str | None = None, rfdetr_size: str = 'medium',
                 rfdetr_threshold: float = 0.1, fusion_cfg: dict | None = None, fusion_head: str | None = None):
        checkpoint = torch.load(salsanext, map_location='cpu', weights_only=False)
        self.classes = Classes(checkpoint['classes'])
        self.lidar_cfg = self.classes.config['lidar']
        self.stats = checkpoint['stats']
        self.device = device
        self.model = SalsaNext(len(self.classes))
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device).eval()

        self.fusion_cfg = {**self.classes.config.get('fusion', {}), **(fusion_cfg or {})}
        weights = self.fusion_cfg.get('weights', {})
        self.lidar_weight = np.asarray(weights.get('lidar', 1.0), dtype=np.float32)
        self.camera_weight = np.asarray(weights.get('camera', 1.0), dtype=np.float32)
        self.method = self.fusion_cfg.get('method', 'product')
        # A trained FusionHead (scripts/train_fusion.py) replaces the fixed rule; its cameras are matched by name.
        self.head = load_head(fusion_head, device) if fusion_head else None

        self.detector = None
        self.rfdetr_threshold = rfdetr_threshold
        if rfdetr:
            import rfdetr as rfdetr_lib
            self.detector = getattr(rfdetr_lib, RFDETR_SIZES[rfdetr_size])(pretrain_weights=rfdetr)
        self._detections: OrderedDict = OrderedDict()

    @torch.no_grad()
    def lidar_probs(self, points: np.ndarray):
        """(N, C) SalsaNext probabilities per point and which points landed on the range image."""
        proj = project(points, self.lidar_cfg['layers_deg'], self.lidar_cfg['width'],
                       self.lidar_cfg.get('layer_tolerance_deg', 0.6))
        image = torch.from_numpy(normalize(proj.image, proj.mask, self.stats))[None].to(self.device)
        probs_img = self.model(image).softmax(1)[0].float().cpu().numpy()
        return lidar_point_probs(probs_img, proj.row, proj.col)

    def detections(self, view: CameraView):
        if view.key not in self._detections:
            self._detections[view.key] = self.detector.predict(Image.fromarray(view.rgb),
                                                               threshold=self.rfdetr_threshold)
            while len(self._detections) > 8:
                self._detections.popitem(last=False)
        return self._detections[view.key]

    def __call__(self, points: np.ndarray, views: list[CameraView] = ()) -> ScanResult:
        p_lidar, lidar_valid = self.lidar_probs(points)
        camera, used = {}, []
        if self.detector is not None:
            for view in views:
                uv, visible = project_to_image(
                    apply(view.optical_from_lidar, points), view.K, view.width, view.height,
                    occlusion_cell=self.fusion_cfg.get('occlusion_cell', 8),
                    occlusion_tol=self.fusion_cfg.get('occlusion_tol', (0.05, 0.3)))
                camera[view.key[0]] = (camera_point_probs(self.detections(view), uv, visible, self.classes), visible)
                used.append(view.key[0])
        if self.head is not None:
            probs, seen = learned_fuse(self.head, p_lidar, lidar_valid, camera, points)
        else:
            probs, seen = fuse(p_lidar, lidar_valid, list(camera.values()), self.method, self.lidar_weight,
                               self.camera_weight)
        labels = np.where(seen, probs.argmax(1), self.classes.ignore).astype(np.uint8)
        confidence = np.where(seen, probs.max(1), 0.0).astype(np.float32)
        return ScanResult(labels, confidence, used)


def cluster_instances(xy: np.ndarray, labels: np.ndarray, class_ids, cell: float = 0.3,
                      min_points: int = 3) -> np.ndarray:
    """Per-scan object ids (1, 2, ...; 0 = none) for the points of `class_ids`: connected cells of a
    bird's-eye grid, separately per class. Ids are not tracked: they restart with every scan."""
    from scipy import ndimage

    ids = np.zeros(len(labels), dtype=np.uint16)
    next_id = 1
    for cid in class_ids:
        members = np.flatnonzero(labels == cid)
        if len(members) == 0:
            continue
        cells = np.floor(xy[members] / cell).astype(np.int64)
        cells -= cells.min(0)
        grid = np.zeros(cells.max(0) + 1, dtype=bool)
        grid[cells[:, 0], cells[:, 1]] = True
        components, count = ndimage.label(grid, structure=np.ones((3, 3), dtype=bool))
        point_component = components[cells[:, 0], cells[:, 1]]
        sizes = np.bincount(point_component, minlength=count + 1)
        for component in range(1, count + 1):
            if sizes[component] < min_points:
                continue
            ids[members[point_component == component]] = next_id
            next_id += 1
    return ids
