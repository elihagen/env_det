#!/usr/bin/env python3
"""Late fusion of SalsaNext (lidar) and RF-DETR-Seg (camera): one class per lidar point.

    python scripts/infer_fusion.py --salsanext outputs/salsanext/best.pt \
        --rfdetr outputs/rfdetr/<best checkpoint> --split test

One dataloader yields each scan with the camera frames nearest in time. The range image goes
through SalsaNext, the frames through RF-DETR; both give class probabilities per point (see
lero_seg/fusion.py), which are merged by a fixed rule (configs/classes.yaml 'fusion', or
--method/--*-weight) or, with --head, by a fusion head trained with scripts/train_fusion.py.

Writes <out>/<recording name>/<frame>.npy (fused train id per point, 255 = no sensor saw it;
--save-probs adds <frame>_probs.npy), preview/ (camera frame with points by fused class | by
ground truth) and metrics.json: per-point IoU of lidar-only, camera-only and fused predictions,
over all points and over the points a camera sees.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rfdetr
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lero_seg.classes import DEFAULT_CONFIG, Classes  # noqa: E402
from lero_seg.fusion import camera_point_probs, fuse, learned_fuse, lidar_point_probs, load_head  # noqa: E402
from lero_seg.fusion_dataset import FusionDataset, collate  # noqa: E402
from lero_seg.metrics import Confusion  # noqa: E402
from lero_seg.recordings import load_split  # noqa: E402
from lero_seg.salsanext import SalsaNext  # noqa: E402

from infer_salsanext import recording_name  # noqa: E402
from train_rfdetr import SIZES  # noqa: E402


def weight(value):
    """One number, or comma-separated per-class weights."""
    parts = [float(v) for v in value.split(',')]
    return parts[0] if len(parts) == 1 else np.array(parts)


def predict(pred_probs, seen, ignore):
    return np.where(seen, pred_probs.argmax(1), ignore).astype(np.uint8)


def scan_probs(batch, salsanext, detector, classes, threshold, device):
    """Per scan of a FusionDataset batch: (index in batch, lidar probs (N, C), which points projected,
    {camera name: (probs (N, C), visible (N,))})."""
    lidar_probs = salsanext(batch['range_image'].to(device)).softmax(1).cpu().numpy()
    # Both lidars' scans can pair with the same camera frame: detect each frame once.
    frames = {(v['camera_dir'], v['frame']): v['rgb'] for views in batch['views'] for v in views}
    detections = detector.predict([Image.fromarray(rgb) for rgb in frames.values()], threshold=threshold) \
        if frames else []
    detections = dict(zip(frames, detections if isinstance(detections, list) else [detections]))
    for b in range(len(batch['labels'])):
        p_lidar, lidar_valid = lidar_point_probs(lidar_probs[b], batch['row'][b], batch['col'][b])
        cameras = {v['camera']: (camera_point_probs(detections[(v['camera_dir'], v['frame'])], v['uv'], v['visible'],
                                                    classes), v['visible']) for v in batch['views'][b]}
        yield b, p_lidar, lidar_valid, cameras


def paint(rgb, uv, visible, classes_per_point, palette, radius=2):
    out = rgb.copy()
    h, w = out.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            u, v = np.clip(uv[visible, 0] + dx, 0, w - 1), np.clip(uv[visible, 1] + dy, 0, h - 1)
            out[v, u] = palette[classes_per_point[visible]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--salsanext', type=Path, required=True, help='SalsaNext checkpoint (best.pt)')
    parser.add_argument('--rfdetr', required=True, help='RF-DETR-Seg checkpoint')
    parser.add_argument('--rfdetr-size', choices=SIZES, default='medium', help='the size the weights were trained as')
    parser.add_argument('--split', choices=('train', 'valid', 'test'),
                        help="score that split of SalsaNext's split.json (next to its checkpoint)")
    parser.add_argument('--recordings', nargs='+', type=Path, help='or: these recording dirs')
    parser.add_argument('--out', type=Path, help='default: <salsanext dir>/fusion_<split or custom>')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help="its 'fusion' section is used")
    parser.add_argument('--cameras', nargs='+', default=['main', 'surround'])
    parser.add_argument('--method', choices=('product', 'mean'), help='default: from the config')
    parser.add_argument('--lidar-weight', type=weight, help='default: from the config')
    parser.add_argument('--camera-weight', type=weight, help='default: from the config')
    parser.add_argument('--head', type=Path, help='fusion head from train_fusion.py (best.pt); replaces --method')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='RF-DETR confidence threshold; low, as confidences are fused, not cut')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-probs', action='store_true', help='also save fused (N, C) float16 probabilities')
    parser.add_argument('--preview-every', type=int, default=20)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not (args.split or args.recordings):
        parser.error('give --split or --recordings')

    checkpoint = torch.load(args.salsanext, map_location='cpu', weights_only=False)
    classes = Classes(checkpoint['classes'])
    fusion_cfg = Classes.load(args.config).config['fusion']
    method = args.method or fusion_cfg['method']
    lidar_weight = args.lidar_weight if args.lidar_weight is not None else np.asarray(fusion_cfg['weights']['lidar'])
    camera_weight = args.camera_weight if args.camera_weight is not None else np.asarray(fusion_cfg['weights']['camera'])
    recordings = args.recordings or load_split(args.salsanext.parent / 'split.json')[args.split]
    out = args.out or args.salsanext.parent / f'fusion_{args.split or "custom"}'
    (out / 'preview').mkdir(parents=True, exist_ok=True)
    palette = classes.palette()

    salsanext = SalsaNext(len(classes))
    salsanext.load_state_dict(checkpoint['model'])
    salsanext.to(args.device).eval()
    detector = getattr(rfdetr, SIZES[args.rfdetr_size])(pretrain_weights=args.rfdetr)
    head = load_head(args.head, args.device) if args.head else None
    if head is not None:
        method = 'learned'

    dataset = FusionDataset(recordings, classes, checkpoint['stats'], fusion_cfg, args.cameras)
    loader = DataLoader(dataset, args.batch_size, num_workers=args.workers, collate_fn=collate)
    scores = {f'{model}_{subset}': Confusion(len(classes), classes.ignore)
              for model in ('lidar', 'camera', 'fused') for subset in ('all', 'camera_view')}
    n_scan = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            for b, p_lidar, lidar_valid, cameras in scan_probs(batch, salsanext, detector, classes, args.threshold,
                                                               args.device):
                labels = batch['labels'][b]
                views = list(cameras.values())
                if head is not None:
                    p_fused, seen = learned_fuse(head, p_lidar, lidar_valid, cameras, batch['points'][b])
                else:
                    p_fused, seen = fuse(p_lidar, lidar_valid, views, method, lidar_weight, camera_weight)
                p_camera, camera_seen = fuse(p_lidar, np.zeros_like(lidar_valid), views, 'product')
                in_view = np.zeros_like(seen) if not views else np.logical_or.reduce([v for _, v in views])

                preds = {'lidar': predict(p_lidar, lidar_valid, classes.ignore),
                         'camera': predict(p_camera, camera_seen, classes.ignore),
                         'fused': predict(p_fused, seen, classes.ignore)}
                for model, pred in preds.items():
                    scores[f'{model}_all'].update(pred, labels)
                    scores[f'{model}_camera_view'].update(pred[in_view], labels[in_view])

                target = out / recording_name(Path(batch['recording'][b]))
                target.mkdir(parents=True, exist_ok=True)
                np.save(target / f'{batch["frame"][b]}.npy', preds['fused'])
                if args.save_probs:
                    np.save(target / f'{batch["frame"][b]}_probs.npy', p_fused.astype(np.float16))
                if n_scan % args.preview_every == 0 and batch['views'][b]:
                    v = batch['views'][b][0]
                    row = np.concatenate([paint(v['rgb'], v['uv'], v['visible'], preds['fused'], palette),
                                          paint(v['rgb'], v['uv'], v['visible'] & (labels != classes.ignore),
                                                labels.astype(np.uint8), palette)], axis=1)
                    Image.fromarray(row).save(out / 'preview' / f'{recording_name(Path(batch["recording"][b])).replace("/", "_")}'
                                                                f'_{batch["frame"][b]}.png')
                n_scan += 1

    report = {'salsanext': str(args.salsanext), 'rfdetr': args.rfdetr, 'scans': len(dataset), 'method': method,
              'head': str(args.head) if args.head else None,
              'lidar_weight': np.asarray(lidar_weight).tolist(), 'camera_weight': np.asarray(camera_weight).tolist(),
              'threshold': args.threshold, **{name: c.report(classes.names) for name, c in scores.items()}}
    with open(out / 'metrics.json', 'w') as f:
        json.dump(report, f, indent=1)
    for name, c in scores.items():
        r = c.report(classes.names)
        print(f'{name:20s} mIoU {r["miou"] if r["miou"] is None else round(r["miou"], 4)}  '
              f'person {r["iou"].get("person")}')
    print(f'predictions, previews + metrics in {out}')


if __name__ == '__main__':
    main()
