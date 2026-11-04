"""Confusion-matrix IoU. An extra last column counts unpredicted (ignore) outputs as misses."""
import numpy as np


class Confusion:
    def __init__(self, num_classes: int, ignore: int = 255):
        self.n, self.ignore = num_classes, ignore
        self.matrix = np.zeros((num_classes, num_classes + 1), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = pred.reshape(-1).astype(np.int64), gt.reshape(-1).astype(np.int64)
        keep = gt != self.ignore
        pred, gt = pred[keep], gt[keep]
        pred = np.where((pred >= 0) & (pred < self.n), pred, self.n)
        self.matrix += np.bincount(gt * (self.n + 1) + pred, minlength=self.n * (self.n + 1)).reshape(self.n, -1)

    def iou(self) -> np.ndarray:
        """Per-class IoU; NaN for classes absent from both ground truth and predictions."""
        tp = np.diag(self.matrix[:, :self.n]).astype(np.float64)
        union = self.matrix.sum(1) + self.matrix[:, :self.n].sum(0) - tp
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(union > 0, tp / union, np.nan)

    def report(self, names) -> dict:
        iou = self.iou()
        total = self.matrix.sum()
        return {'miou': float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
                'accuracy': float(np.trace(self.matrix[:, :self.n]) / total) if total else None,
                'iou': {name: (None if np.isnan(v) else round(float(v), 4)) for name, v in zip(names, iou)},
                'gt_count': {name: int(c) for name, c in zip(names, self.matrix.sum(1))}}
