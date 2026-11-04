#!/usr/bin/env python3
"""Predict per-point classes with a trained SalsaNext and score them against the simulator labels.

    python scripts/infer_salsanext.py --checkpoint outputs/salsanext/best.pt --split test
    python scripts/infer_salsanext.py --checkpoint ... --recordings REC_DIR [REC_DIR ...]

Writes <out>/<recording name>/<frame>.npy (train ids per point, 255 = not projected) and
metrics.json (per-point IoU over all scored scans).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lero_seg.classes import Classes  # noqa: E402
from lero_seg.lidar_dataset import RangeImageDataset, normalize  # noqa: E402
from lero_seg.metrics import Confusion  # noqa: E402
from lero_seg.recordings import load_split  # noqa: E402
from lero_seg.salsanext import SalsaNext  # noqa: E402


def recording_name(recording: Path) -> str:
    """run/scene, e.g. run_20261103_1407/scene_0003 (the recording dir itself is 'recording')."""
    parts = recording.parts
    return '/'.join(parts[-3:-1]) if parts[-1] == 'recording' else recording.name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'valid', 'test'),
                        help="score that split of the training run's split.json (next to the checkpoint)")
    parser.add_argument('--recordings', nargs='+', type=Path, help='or: these recording dirs')
    parser.add_argument('--out', type=Path, help='default: <checkpoint dir>/pred_<split or custom>')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not (args.split or args.recordings):
        parser.error('give --split or --recordings')

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    classes = Classes(checkpoint['classes'])
    recordings = args.recordings or load_split(args.checkpoint.parent / 'split.json')[args.split]
    out = args.out or args.checkpoint.parent / f'pred_{args.split or "custom"}'
    model = SalsaNext(len(classes))
    model.load_state_dict(checkpoint['model'])
    model.to(args.device).eval()

    dataset = RangeImageDataset(recordings, classes, classes.config['lidar'], stats=checkpoint['stats'])
    confusion = Confusion(len(classes), classes.ignore)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            proj, _, _, labels = dataset.project(i)
            image = torch.from_numpy(normalize(proj.image, proj.mask, dataset.stats))[None].to(args.device)
            pred_img = model(image).argmax(1)[0].cpu().numpy().astype(np.uint8)
            # Every projected point takes its pixel's class (also points that lost the pixel to a nearer one).
            pred = np.full(len(labels), classes.ignore, dtype=np.uint8)
            hit = proj.row >= 0
            pred[hit] = pred_img[proj.row[hit], proj.col[hit]]
            confusion.update(pred, labels)
            item = dataset.items[i]
            target = out / recording_name(item['recording'])
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / f'{item["frame"]}.npy', pred)

    report = {'checkpoint': str(args.checkpoint), 'scans': len(dataset), **confusion.report(classes.names)}
    with open(out / 'metrics.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ('miou', 'accuracy', 'iou')}, indent=1))
    print(f'predictions + metrics in {out}')


if __name__ == '__main__':
    main()
