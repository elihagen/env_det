"""Training class table and the mappings from the recorder's lidar ids / camera labels to it."""
import colorsys
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / 'configs' / 'classes.yaml'


class Classes:
    def __init__(self, config: dict):
        self.config = config
        self.names = list(config['classes'])
        self.ignore = int(config['ignore_index'])
        self.to_train = {k: self.names.index(v) for k, v in config['recorder_to_train'].items()}
        self._longest_first = sorted(self.to_train, key=len, reverse=True)
        self.things = [self.names.index(n) for n in config.get('things', [])]

    @classmethod
    def load(cls, path=DEFAULT_CONFIG) -> 'Classes':
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def __len__(self) -> int:
        return len(self.names)

    def from_camera_label(self, label: str) -> int:
        first = label.split(',')[0].strip()
        if first in self.to_train:
            return self.to_train[first]
        return next((self.to_train[n] for n in self._longest_first if n in label), self.ignore)

    def camera_lut(self, labels: dict[str, str], max_id: int) -> np.ndarray:
        """Lookup table semantic-image id -> train id for one frame's id -> label map."""
        lut = np.full(max(max_id, *(int(k) for k in labels), 0) + 1, self.ignore, dtype=np.uint8)
        for key, label in labels.items():
            lut[int(key)] = self.from_camera_label(label)
        return lut

    def lidar_lut(self, recorder_classes: dict[str, str]) -> np.ndarray:
        """Lookup table recorder lidar class id (classes.json) -> train id."""
        lut = np.full(256, self.ignore, dtype=np.uint8)
        for key, name in recorder_classes.items():
            if name in self.to_train:
                lut[int(key)] = self.to_train[name]
        return lut

    def palette(self) -> np.ndarray:
        """(256, 3) uint8 colours: one hue per class, ignore = black."""
        out = np.zeros((256, 3), dtype=np.uint8)
        for i in range(len(self)):
            out[i] = [int(255 * c) for c in colorsys.hsv_to_rgb(i / len(self), 0.75, 0.95)]
        return out
