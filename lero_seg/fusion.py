"""Late fusion of SalsaNext (lidar) and RF-DETR-Seg (camera) class probabilities, per lidar point.

SalsaNext gives a softmax per range-image pixel; each point takes its pixel's. RF-DETR gives
instance masks with a confidence each; a point's camera score for a class is the highest
confidence among that class's masks covering its pixel. The scores become a distribution that
keeps (1 - best score) spread over all classes, so a point no mask covers carries no camera
evidence. Points the camera doesn't see (outside the image or occluded) keep the lidar's.

The probabilities are merged either by a fixed rule (fuse: 'product' or 'mean') or by a trained
FusionHead (learned_fuse; trained by scripts/train_fusion.py).
"""
import numpy as np
import torch
from torch import nn

EPS = 1e-6


def lidar_point_probs(probs_img: np.ndarray, row: np.ndarray, col: np.ndarray):
    """(N, C) per-point probabilities from SalsaNext's (C, H, W) softmax, and which points projected."""
    valid = row >= 0
    probs = np.full((len(row), probs_img.shape[0]), 1.0 / probs_img.shape[0], dtype=np.float32)
    probs[valid] = probs_img[:, row[valid], col[valid]].T
    return probs, valid


def detection_train_ids(detections, classes) -> np.ndarray:
    """Train id per detection, -1 for unknown classes (export_coco.py's category ids are train id + 1)."""
    names = detections.data.get('class_name')
    if names is not None:
        return np.array([classes.names.index(n) if n in classes.names else -1 for n in names], dtype=np.int64)
    ids = np.asarray(detections.class_id, dtype=np.int64) - 1
    return np.where((ids >= 0) & (ids < len(classes)), ids, -1)


def camera_point_probs(detections, uv: np.ndarray, visible: np.ndarray, classes) -> np.ndarray:
    """(N, C) per-point probabilities from RF-DETR's masks at the points' pixels (uniform where none)."""
    scores = np.zeros((len(uv), len(classes)), dtype=np.float32)
    seen = np.flatnonzero(visible)
    if detections.mask is not None and len(detections) and len(seen):
        for i, cid in enumerate(detection_train_ids(detections, classes)):
            if cid < 0:
                continue
            hit = seen[detections.mask[i][uv[seen, 1], uv[seen, 0]]]
            scores[hit, cid] = np.maximum(scores[hit, cid], detections.confidence[i])
    return scores_to_probs(scores)


def scores_to_probs(scores: np.ndarray) -> np.ndarray:
    best = scores.max(1, keepdims=True)
    total = scores.sum(1, keepdims=True)
    shares = np.divide(scores, total, out=np.zeros_like(scores), where=total > 0)
    return shares * best + (1 - best) / scores.shape[1]


def fuse(lidar_probs, lidar_valid, camera_views, method='product', lidar_weight=1.0, camera_weight=1.0):
    """Fused (N, C) probabilities and whether any sensor saw the point.

    camera_views: [(probs (N, C), visible (N,))], one per camera. Weights are scalars or (C,) arrays.
    """
    w_lidar = np.asarray(lidar_weight, dtype=np.float32) * lidar_valid[:, None]
    experts = [(lidar_probs, w_lidar)]
    experts += [(probs, np.asarray(camera_weight, dtype=np.float32) * visible[:, None]) for probs, visible in camera_views]
    seen = lidar_valid.copy()
    for _, visible in camera_views:
        seen |= visible
    if method == 'product':
        log_p = sum(w * np.log(np.clip(p, EPS, 1.0)) for p, w in experts)
        log_p -= log_p.max(1, keepdims=True)
        fused = np.exp(log_p)
    elif method == 'mean':
        fused = sum(w * p for p, w in experts)
    else:
        raise ValueError(f'unknown fusion method {method!r}')
    total = fused.sum(1, keepdims=True)
    uniform = np.full_like(fused, 1.0 / fused.shape[1])
    return np.divide(fused, total, out=uniform, where=total > 0), seen


# Point geometry fed to the FusionHead: (range, z) in the lidar frame, divided by these [m].
GEOMETRY_SCALE = (20.0, 5.0)


def point_geometry(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float32)[:, :3]
    return np.stack([np.linalg.norm(xyz, axis=1) / GEOMETRY_SCALE[0], xyz[:, 2] / GEOMETRY_SCALE[1]], 1)


def stack_sources(lidar_probs, lidar_valid, cameras: dict, camera_order) -> tuple[np.ndarray, np.ndarray]:
    """(N, S, C) probabilities and (N, S) validity of the S = 1 + len(camera_order) sources (lidar
    first, then the cameras in camera_order). cameras: {name: (probs (N, C), visible (N,))}; a camera
    that is missing (or not in camera_order) counts as not seeing any point."""
    n, c = lidar_probs.shape
    probs = np.full((n, 1 + len(camera_order), c), 1.0 / c, dtype=np.float32)
    valid = np.zeros((n, 1 + len(camera_order)), dtype=bool)
    probs[:, 0], valid[:, 0] = lidar_probs, lidar_valid
    for i, name in enumerate(camera_order, 1):
        if name in cameras:
            probs[:, i], valid[:, i] = cameras[name]
    return probs, valid


class FusionHead(nn.Module):
    """Trained late fusion, per point: an MLP of `layers` linear layers over every source's
    log-probabilities and validity (lidar, then each camera) plus the point's geometry, whose output
    is added to the sources' summed log-probabilities. The last layer starts at zero, so an untrained
    head equals the 'product' rule with weight 1; training learns the correction (per-class trust,
    confidence calibration, class interactions between the sensors)."""

    def __init__(self, num_classes: int, cameras, hidden: int = 64, layers: int = 3, geometry: bool = True):
        super().__init__()
        self.config = {'num_classes': num_classes, 'cameras': list(cameras), 'hidden': hidden, 'layers': layers,
                       'geometry': geometry}
        self.cameras, self.geometry = list(cameras), geometry
        sources = 1 + len(self.cameras)
        dims = [sources * (num_classes + 1) + (2 if geometry else 0)] + [hidden] * (layers - 1) + [num_classes]
        modules = []
        for i in range(layers):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < layers - 1:
                modules += [nn.LayerNorm(dims[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*modules)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, probs, valid, geometry=None):
        """Logits (N, C) from probs (N, S, C), valid (N, S) and geometry (N, 2)."""
        valid = valid.float()
        log_p = probs.float().clamp(EPS, 1.0).log() * valid[..., None]
        features = [torch.cat([log_p, valid[..., None]], -1).flatten(1)]
        if self.geometry:
            features.append(geometry.float())
        return log_p.sum(1) + self.mlp(torch.cat(features, 1))


def load_head(path, device='cpu') -> FusionHead:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    head = FusionHead(**checkpoint['config'])
    head.load_state_dict(checkpoint['model'])
    return head.to(device).eval()


@torch.no_grad()
def learned_fuse(head: FusionHead, lidar_probs, lidar_valid, cameras: dict, points):
    """Like fuse(), with a FusionHead. cameras: {name: (probs (N, C), visible (N,))}."""
    probs, valid = stack_sources(lidar_probs, lidar_valid, cameras, head.cameras)
    device = next(head.parameters()).device
    logits = head(torch.from_numpy(probs).to(device), torch.from_numpy(valid).to(device),
                  torch.from_numpy(point_geometry(points)).to(device))
    fused = logits.softmax(1).cpu().numpy()
    seen = valid.any(1)
    fused[~seen] = 1.0 / fused.shape[1]
    return fused, seen
