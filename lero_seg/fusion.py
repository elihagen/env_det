"""Late fusion of SalsaNext (lidar) and RF-DETR-Seg (camera) class probabilities, per lidar point.

SalsaNext gives a softmax per range-image pixel; each point takes its pixel's. RF-DETR gives
instance masks with a confidence each; a point's camera score for a class is the highest
confidence among that class's masks covering its pixel. The scores become a distribution that
keeps (1 - best score) spread over all classes, so a point no mask covers carries no camera
evidence. Points the camera doesn't see (outside the image or occluded) keep the lidar's.
"""
import numpy as np

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
