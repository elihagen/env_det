#!/usr/bin/env python3
"""Predict per-point classes with a trained SalsaNext and score them against the simulator labels.

    python scripts/infer_salsanext.py --checkpoint outputs/salsanext/best.pt --split test
    python scripts/infer_salsanext.py --checkpoint ... --recordings REC_DIR [REC_DIR ...] --ply

Also runs on recordings without point_labels (new scenes): they're predicted and previewed, not scored.
Writes <out>/<recording name>/<frame>.npy (train ids per point, 255 = not projected),
preview/ (every --preview-every-th scan: range image and bird's-eye view, ground truth | prediction),
with --ply <frame>.ply point clouds coloured by prediction, and metrics.json (per-point IoU).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lero_seg.classes import Classes  # noqa: E402
from lero_seg.lidar_dataset import RangeImageDataset, normalize  # noqa: E402
from lero_seg.metrics import Confusion  # noqa: E402
from lero_seg.recordings import load_split  # noqa: E402
from lero_seg.salsanext import SalsaNext  # noqa: E402
from lero_seg.visualize import bev, legend, range_strip, save_ply, title  # noqa: E402


def recording_name(recording: Path) -> str:
    """run/scene, e.g. run_20261103_1407/scene_0003 (the recording dir itself is 'recording')."""
    parts = recording.parts
    return '/'.join(parts[-3:-1]) if parts[-1] == 'recording' else recording.name


def preview(points, labels, pred, label_img, pred_img, mask, item, classes, palette) -> np.ndarray:
    """Range image (ground truth over prediction) above bird's-eye views (ground truth | prediction)."""
    gt_name = 'ground truth' if item['labels'] is not None else 'ground truth (none recorded)'
    views = np.concatenate([title(bev(points, labels, palette), gt_name), np.full((500, 4, 3), 255, np.uint8),
                            title(bev(points, pred, palette), 'prediction'),
                            legend(classes.names, palette, 500)], axis=1)
    strips = np.concatenate([range_strip(label_img.clip(0, 255).astype(np.uint8), mask, palette),
                             np.full((4, mask.shape[1], 3), 255, np.uint8),
                             range_strip(pred_img, mask, palette)], axis=0)
    strips = np.pad(strips, ((0, 0), (0, max(0, views.shape[1] - strips.shape[1])), (0, 0)))[:, :views.shape[1]]
    header = title(np.zeros((30, views.shape[1], 3), np.uint8),
                   f'{recording_name(item["recording"])}  frame {item["frame"]}  {item["topic"]}  '
                   '(range image: ground truth above prediction)')
    return np.concatenate([header, strips, np.zeros((6, views.shape[1], 3), np.uint8), views], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'valid', 'test'),
                        help="score that split of the training run's split.json (next to the checkpoint)")
    parser.add_argument('--recordings', nargs='+', type=Path, help='or: these recording dirs')
    parser.add_argument('--out', type=Path, help='default: <checkpoint dir>/pred_<split or custom>')
    parser.add_argument('--preview-every', type=int, default=10, help='preview every n-th scan (0: none)')
    parser.add_argument('--ply', action='store_true', help='also write each scan as a coloured .ply')
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

    dataset = RangeImageDataset(recordings, classes, classes.config['lidar'], stats=checkpoint['stats'],
                                require_labels=False)
    confusion = Confusion(len(classes), classes.ignore)
    palette = classes.palette()
    (out / 'preview').mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            proj, label_img, points, labels = dataset.project(i)
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
            if args.ply:
                save_ply(target / f'{item["frame"]}.ply', points, palette[pred])
            if args.preview_every and i % args.preview_every == 0:
                name = f'{recording_name(item["recording"]).replace("/", "_")}_{item["frame"]}.png'
                Image.fromarray(preview(points, labels, pred, label_img, pred_img, proj.mask, item, classes,
                                        palette)).save(out / 'preview' / name)

    report = {'checkpoint': str(args.checkpoint), 'scans': len(dataset), **confusion.report(classes.names)}
    with open(out / 'metrics.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ('miou', 'accuracy', 'iou')}, indent=1))
    print(f'predictions + metrics in {out}')


if __name__ == '__main__':
    main()
