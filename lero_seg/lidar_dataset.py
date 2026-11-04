"""Torch dataset of projected lidar scans (each lidar's scan is its own sample)."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .range_image import lidar_frames, load_scan, project


class RangeImageDataset(Dataset):
    def __init__(self, recordings, classes, lidar_cfg, stats=None, augment=False):
        self.items = [item for rec in recordings for item in lidar_frames(rec)]
        self.classes, self.cfg, self.augment = classes, lidar_cfg, augment
        self.stats = stats  # {'mean': [5], 'std': [5]}, set by compute_stats() for training

    def __len__(self):
        return len(self.items)

    def project(self, idx):
        points, labels = load_scan(self.items[idx], self.classes)
        proj = project(points, self.cfg['layers_deg'], self.cfg['width'], self.cfg.get('layer_tolerance_deg', 0.6))
        label_img = np.full(proj.mask.shape, self.classes.ignore, dtype=np.int64)
        label_img[proj.mask] = labels[proj.index[proj.mask]]
        return proj, label_img, points, labels

    def __getitem__(self, idx):
        proj, label_img, _, _ = self.project(idx)
        image, mask = proj.image, proj.mask
        if self.augment:
            shift = np.random.randint(image.shape[2])  # rotate about the vertical axis
            image, mask, label_img = (np.roll(a, shift, axis=-1) for a in (image, mask, label_img))
            if np.random.rand() < 0.5:  # mirror left/right: reverse azimuth, negate y
                image, mask, label_img = (a[..., ::-1] for a in (image, mask, label_img))
                image = image.copy()
                image[2] *= -1
        image = normalize(image, mask, self.stats)
        return (torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(label_img)),
                torch.from_numpy(np.ascontiguousarray(mask)))

    def compute_stats(self, max_scans=300):
        """Per-channel mean/std over occupied pixels of up to `max_scans` scans."""
        picks = np.linspace(0, len(self) - 1, min(max_scans, len(self))).astype(int)
        projections = [self.project(i)[0] for i in picks]
        values = np.concatenate([p.image[:, p.mask] for p in projections], axis=1)
        self.stats = {'mean': values.mean(1).tolist(), 'std': (values.std(1) + 1e-6).tolist()}
        return self.stats

    def class_frequencies(self, max_scans=300):
        picks = np.linspace(0, len(self) - 1, min(max_scans, len(self))).astype(int)
        counts = np.zeros(len(self.classes), dtype=np.int64)
        for i in picks:
            labels = self.project(i)[1]
            labels = labels[labels != self.classes.ignore]
            counts += np.bincount(labels, minlength=len(self.classes))
        return counts


def normalize(image, mask, stats):
    if stats is None:
        return image.astype(np.float32)
    mean = np.asarray(stats['mean'], dtype=np.float32)[:, None, None]
    std = np.asarray(stats['std'], dtype=np.float32)[:, None, None]
    return ((image - mean) / std * mask[None]).astype(np.float32)  # empty pixels stay 0
