"""Reading a recorded camera's frames: rgb, semantic ids as train ids, and instance ids."""
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .recordings import read_jsonl

# Instance paths of one placed object share this prefix (a tree's meshes, a person's body parts).
OBJECT_ROOT = re.compile(r'^/World/(?:Objects|Border)/[^/]+')


def camera_frames(cam_dir: Path) -> list[dict]:
    """camera_index.jsonl entries, with 'instance_paths' from persons.jsonl where recorded."""
    frames = read_jsonl(cam_dir / 'camera_index.jsonl')
    persons = cam_dir / 'persons.jsonl'
    if persons.is_file():
        paths = {e['frame']: e.get('instance_paths', {}) for e in read_jsonl(persons)}
        for entry in frames:
            entry['instance_paths'] = paths.get(entry['frame'])
    return frames


def rgb_path(cam_dir: Path, entry: dict) -> Path:
    return cam_dir / 'rgb' / f'{entry["frame"]}.png'


def semantic_train_ids(cam_dir: Path, entry: dict, classes) -> np.ndarray:
    ids = np.array(Image.open(cam_dir / 'semantic' / f'{entry["frame"]}.png')).astype(np.int64)
    return classes.camera_lut(entry['labels'], int(ids.max()))[ids]


def instance_ids(cam_dir: Path, entry: dict) -> np.ndarray | None:
    for suffix in ('png', 'npy'):
        path = cam_dir / 'instance' / f'{entry["frame"]}.{suffix}'
        if path.is_file():
            return np.load(path) if suffix == 'npy' else np.array(Image.open(path)).astype(np.int64)
    return None


def object_masks(instances: np.ndarray, instance_paths: dict, region: np.ndarray) -> list[np.ndarray]:
    """Masks of the separate objects inside `region` (e.g. all person pixels), one per placed object."""
    groups: dict[str, list[int]] = {}
    for key, path in instance_paths.items():
        match = OBJECT_ROOT.match(path)
        if key.isdigit():
            groups.setdefault(match.group(0) if match else path, []).append(int(key))
    present = set(np.unique(instances[region]).tolist())
    masks = []
    for ids in groups.values():
        ids = [i for i in ids if i in present]
        if ids:
            mask = region & np.isin(instances, ids)
            if mask.any():
                masks.append(mask)
    return masks
