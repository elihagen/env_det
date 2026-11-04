"""Finding recordings (one per datagen scene) and splitting them into train/valid/test.

Splits are by scene, never by frame: frames of one 4 s drive are nearly identical.
"""
import json
import random
from pathlib import Path

SPLITS = ('train', 'valid', 'test')


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_recordings(roots) -> list[Path]:
    """Every recording directory (holding lidar_index.jsonl or camera_index.jsonl) under `roots`."""
    found = set()
    for root in roots:
        for name in ('lidar_index.jsonl', 'camera_index.jsonl'):
            for index in Path(root).rglob(name):
                if index.parent.name != 'surround':
                    found.add(index.parent.resolve())
    return sorted(found)


def split_recordings(recordings, valid_frac=0.1, test_frac=0.1, seed=0) -> dict[str, list[Path]]:
    recordings = sorted(recordings)
    random.Random(seed).shuffle(recordings)
    n = len(recordings)
    n_valid = max(1, round(n * valid_frac)) if n >= 3 else 0
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    return {'valid': recordings[:n_valid], 'test': recordings[n_valid:n_valid + n_test],
            'train': recordings[n_valid + n_test:]}


def save_split(split: dict, path: Path) -> None:
    with open(path, 'w') as f:
        json.dump({k: [str(p) for p in v] for k, v in split.items()}, f, indent=1)


def load_split(path: Path) -> dict[str, list[Path]]:
    with open(path) as f:
        return {k: [Path(p) for p in v] for k, v in json.load(f).items()}


def camera_dirs(recording: Path) -> dict[str, Path]:
    """The recording's cameras that saved frames: 'main' (the root) and 'surround'."""
    cams = {'main': recording, 'surround': recording / 'surround'}
    return {name: d for name, d in cams.items() if (d / 'camera_index.jsonl').is_file()}


def recorder_classes(recording: Path) -> dict[str, str]:
    with open(recording / 'classes.json') as f:
        return json.load(f)
