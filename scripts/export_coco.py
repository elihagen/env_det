#!/usr/bin/env python3
"""Export recorded camera frames as a COCO instance-segmentation dataset for RF-DETR-Seg.

    python scripts/export_coco.py --data /path/to/datagen_runs --out datasets/coco_seg

Writes <out>/{train,valid,test}/_annotations.coco.json + images (symlinks to the recordings'
rgb/*.png unless --copy), sources.json (image -> recording frame, for inference scoring) and
split.json. Objects of the 'things' classes (configs/classes.yaml) are one instance each, split
by the recorded instance ids; every other class is one (multi-polygon) region per image.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lero_seg.camera import camera_frames, instance_ids, object_masks, rgb_path, semantic_train_ids  # noqa: E402
from lero_seg.classes import DEFAULT_CONFIG, Classes  # noqa: E402
from lero_seg.recordings import SPLITS, camera_dirs, find_recordings, load_split, save_split, \
    split_recordings  # noqa: E402


def polygons(mask: np.ndarray, min_area: float) -> list[list[float]]:
    """Outer contours of `mask` as COCO polygons (holes are filled - plain COCO polygons have none)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = cv2.approxPolyDP(contour, 1.0, True)
        if len(contour) >= 3:
            out.append(contour.reshape(-1).astype(float).tolist())
    return out


def annotation(mask, category_id, polys, image_id, ann_id):
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return {'id': ann_id, 'image_id': image_id, 'category_id': category_id, 'segmentation': polys,
            'area': int(mask.sum()), 'bbox': [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1],
            'iscrowd': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', nargs='+', required=True, help='folders searched for recordings')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--split', type=Path, help="reuse a split.json (e.g. SalsaNext's) for the same scenes")
    parser.add_argument('--valid-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--every-n', type=int, default=2, help='keep every n-th frame (neighbours are near-identical)')
    parser.add_argument('--cameras', nargs='+', default=['main', 'surround'])
    parser.add_argument('--min-area', type=float, default=20.0, help='drop polygons smaller than this [px]')
    parser.add_argument('--copy', action='store_true', help='copy images instead of symlinking them')
    args = parser.parse_args()

    classes = Classes.load(args.config)
    split = load_split(args.split) if args.split else split_recordings(
        find_recordings(args.data), args.valid_frac, args.test_frac, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    save_split(split, args.out / 'split.json')
    # COCO category ids start at 1; RF-DETR reports fine-tuned classes by these names.
    categories = [{'id': i + 1, 'name': name, 'supercategory': 'thing' if i in classes.things else 'stuff'}
                  for i, name in enumerate(classes.names)]
    sources = {}
    for split_name in SPLITS:
        folder = args.out / split_name
        folder.mkdir(exist_ok=True)
        images, annotations = [], []
        for recording in tqdm(split[split_name], desc=split_name):
            scene = f'{recording.parent.parent.name}_{recording.parent.name}'
            for cam_name, cam_dir in camera_dirs(recording).items():
                if cam_name not in args.cameras:
                    continue
                for entry in camera_frames(cam_dir)[::args.every_n]:
                    semantic = semantic_train_ids(cam_dir, entry, classes)
                    instances = instance_ids(cam_dir, entry) if entry.get('instance_paths') else None
                    image_id = len(images) + 1
                    file_name = f'{scene}_{cam_name}_{entry["frame"]}.png'
                    for cid in np.unique(semantic):
                        if cid == classes.ignore:
                            continue
                        region = semantic == cid
                        if cid in classes.things and instances is not None:
                            masks = object_masks(instances, entry['instance_paths'], region) or [region]
                        else:
                            masks = [region]
                        for mask in masks:
                            polys = polygons(mask, args.min_area)
                            if polys:
                                annotations.append(annotation(mask, int(cid) + 1, polys, image_id,
                                                              len(annotations) + 1))
                    target = folder / file_name
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    source = rgb_path(cam_dir, entry)
                    shutil.copy(source, target) if args.copy else target.symlink_to(source)
                    images.append({'id': image_id, 'file_name': file_name,
                                   'width': int(semantic.shape[1]), 'height': int(semantic.shape[0])})
                    sources[f'{split_name}/{file_name}'] = {'camera_dir': str(cam_dir), 'frame': entry['frame']}
        with open(folder / '_annotations.coco.json', 'w') as f:
            json.dump({'info': {'description': 'LERO Isaac Sim recordings'}, 'licenses': [],
                       'images': images, 'categories': categories, 'annotations': annotations}, f)
        print(f'{split_name}: {len(images)} images, {len(annotations)} instances')
    with open(args.out / 'sources.json', 'w') as f:
        json.dump(sources, f, indent=1)


if __name__ == '__main__':
    main()
