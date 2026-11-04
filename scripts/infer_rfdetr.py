#!/usr/bin/env python3
"""Semantic segmentation with a fine-tuned RF-DETR-Seg, scored against the simulator's labels.

    python scripts/infer_rfdetr.py --weights outputs/rfdetr/<best checkpoint> --dataset datasets/coco_seg --split test

RF-DETR predicts instance masks; they're painted into one class map per image, most
confident last, so overlaps go to the surest instance. Pixels no mask covers count as
misses. Writes <out>/pred/<image>.png (train ids, 255 = none), preview/ (rgb | prediction |
ground truth, every --preview-every-th image) and metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rfdetr
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lero_seg.camera import semantic_train_ids  # noqa: E402
from lero_seg.classes import DEFAULT_CONFIG, Classes  # noqa: E402
from lero_seg.metrics import Confusion  # noqa: E402
from lero_seg.recordings import read_jsonl  # noqa: E402

from train_rfdetr import SIZES  # noqa: E402


def to_semantic(detections, shape, classes, coco_names) -> np.ndarray:
    semantic = np.full(shape, classes.ignore, dtype=np.uint8)
    if detections.mask is None or len(detections) == 0:
        return semantic
    names = detections.data.get('class_name')
    for i in np.argsort(detections.confidence):  # ascending: the most confident is painted last
        name = names[i] if names is not None else coco_names.get(int(detections.class_id[i]))
        if name in classes.names:
            semantic[detections.mask[i]] = classes.names.index(name)
    return semantic


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--size', choices=SIZES, default='medium', help='the size the weights were trained as')
    parser.add_argument('--dataset', type=Path, required=True, help='export_coco.py output')
    parser.add_argument('--split', choices=('train', 'valid', 'test'), default='test')
    parser.add_argument('--out', type=Path, help='default: <weights dir>/pred_<split>')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--preview-every', type=int, default=10)
    args = parser.parse_args()

    classes = Classes.load(args.config)
    palette = classes.palette()
    out = args.out or Path(args.weights).parent / f'pred_{args.split}'
    (out / 'pred').mkdir(parents=True, exist_ok=True)
    (out / 'preview').mkdir(exist_ok=True)
    with open(args.dataset / args.split / '_annotations.coco.json') as f:
        coco = json.load(f)
    coco_names = {c['id']: c['name'] for c in coco['categories']}
    with open(args.dataset / 'sources.json') as f:
        sources = json.load(f)
    model = getattr(rfdetr, SIZES[args.size])(pretrain_weights=args.weights)

    confusion = Confusion(len(classes), classes.ignore)
    labels_cache: dict[str, dict] = {}
    for n, image_info in enumerate(tqdm(coco['images'])):
        image = Image.open(args.dataset / args.split / image_info['file_name']).convert('RGB')
        detections = model.predict(image, threshold=args.threshold)
        pred = to_semantic(detections, (image.height, image.width), classes, coco_names)
        Image.fromarray(pred).save(out / 'pred' / image_info['file_name'])

        source = sources[f'{args.split}/{image_info["file_name"]}']
        cam_dir = Path(source['camera_dir'])
        if source['camera_dir'] not in labels_cache:
            labels_cache[source['camera_dir']] = {e['frame']: e for e in read_jsonl(cam_dir / 'camera_index.jsonl')}
        gt = semantic_train_ids(cam_dir, labels_cache[source['camera_dir']][source['frame']], classes)
        confusion.update(pred, gt)
        if n % args.preview_every == 0:
            row = np.concatenate([np.array(image), palette[pred], palette[gt]], axis=1)
            Image.fromarray(row).save(out / 'preview' / image_info['file_name'])

    report = {'weights': args.weights, 'images': len(coco['images']), 'threshold': args.threshold,
              **confusion.report(classes.names)}
    with open(out / 'metrics.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ('miou', 'accuracy', 'iou')}, indent=1))
    print(f'predictions, previews + metrics in {out}')


if __name__ == '__main__':
    main()
