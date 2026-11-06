#!/usr/bin/env python3
"""Train a fusion head: a small per-point MLP that merges SalsaNext's and RF-DETR-Seg's class
probabilities, trained on the lidar points' semantic labels. Both models stay frozen.

    python scripts/train_fusion.py --salsanext outputs/salsanext/best.pt \
        --rfdetr outputs/rfdetr/<best checkpoint> --split valid

1. Cache: every scan of the split goes through both models once (as in infer_fusion.py); each
   source's per-point probabilities, validity, the point geometry and the labels go to
   <out>/cache/<recording name>/<frame>.npz. Reused on the next run if the models and settings match.
2. Train: FusionHead (lero_seg/fusion.py) with class-weighted cross-entropy on a random subset of
   each scan's labelled points that some sensor sees. Its output is added to the product of the
   sources, so it starts as the 'product' rule and learns the correction.

Train it on scans neither model was trained on, else it learns to trust their overfitted outputs:
by default the 'valid' split of SalsaNext's split.json (export_coco.py uses the same split for
RF-DETR). The last --holdout-frac of each recording's scans (by time) selects the best epoch; score
the head on the test split with infer_fusion.py --head <out>/best.pt.

Writes <out>/best.pt, last.pt and history.json (holdout mIoU of the head and of the product rule).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rfdetr
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lero_seg.classes import DEFAULT_CONFIG, Classes  # noqa: E402
from lero_seg.fusion import FusionHead, point_geometry, stack_sources  # noqa: E402
from lero_seg.fusion_dataset import FusionDataset, collate  # noqa: E402
from lero_seg.metrics import Confusion  # noqa: E402
from lero_seg.recordings import load_split  # noqa: E402
from lero_seg.salsanext import SalsaNext  # noqa: E402

from infer_fusion import scan_probs  # noqa: E402
from infer_salsanext import recording_name  # noqa: E402
from train_rfdetr import SIZES  # noqa: E402


def build_cache(args, classes, checkpoint, fusion_cfg, recordings, cache: Path) -> list[dict]:
    """Run both models over the recordings once; returns one entry per scan (file, recording, stamp)."""
    meta = {'salsanext': str(args.salsanext.resolve()), 'rfdetr': str(Path(args.rfdetr).resolve()),
            'rfdetr_size': args.rfdetr_size, 'threshold': args.threshold, 'cameras': args.cameras,
            'recordings': sorted(str(r) for r in recordings)}
    dataset = FusionDataset(recordings, classes, checkpoint['stats'], fusion_cfg, args.cameras)
    entries = [{'file': cache / recording_name(Path(item['recording'])) / f'{item["frame"]}.npz',
                'recording': str(item['recording']), 'stamp_ns': item['stamp_ns']} for item in dataset.lidar.items]
    meta_file = cache / 'meta.json'
    if meta_file.exists() and json.loads(meta_file.read_text()) == meta and all(e['file'].exists() for e in entries):
        print(f'reusing the cache in {cache}')
        return entries
    if meta_file.exists():
        print(f'cache in {cache} is from other models/settings: rebuilding')
        meta_file.unlink()

    salsanext = SalsaNext(len(classes))
    salsanext.load_state_dict(checkpoint['model'])
    salsanext.to(args.device).eval()
    detector = getattr(rfdetr, SIZES[args.rfdetr_size])(pretrain_weights=args.rfdetr)
    loader = DataLoader(dataset, args.cache_batch_size, num_workers=args.workers, collate_fn=collate)
    with torch.no_grad():
        for batch in tqdm(loader, desc='caching model outputs'):
            for b, p_lidar, lidar_valid, cameras in scan_probs(batch, salsanext, detector, classes, args.threshold,
                                                               args.device):
                probs, valid = stack_sources(p_lidar, lidar_valid, cameras, args.cameras)
                target = cache / recording_name(Path(batch['recording'][b])) / f'{batch["frame"][b]}.npz'
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, probs=probs.astype(np.float16), valid=valid,
                                    geometry=point_geometry(batch['points'][b]).astype(np.float16),
                                    labels=batch['labels'][b].astype(np.uint8))
    meta_file.write_text(json.dumps(meta, indent=1))
    return entries


def holdout_split(entries: list[dict], frac: float) -> tuple[list[Path], list[Path]]:
    """The last `frac` of each recording's scans (by time) are held out: neighbouring scans look alike."""
    by_recording: dict[str, list[dict]] = {}
    for e in entries:
        by_recording.setdefault(e['recording'], []).append(e)
    train, holdout = [], []
    for scans in by_recording.values():
        scans.sort(key=lambda e: e['stamp_ns'])
        cut = len(scans) - int(round(len(scans) * frac))
        train += [e['file'] for e in scans[:cut]]
        holdout += [e['file'] for e in scans[cut:]]
    return train, holdout


class CachedScans(Dataset):
    """Points of cached scans; with `points`, a random subset of the labelled, seen points per scan."""

    def __init__(self, files, ignore, points=None):
        self.files, self.ignore, self.points = files, ignore, points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        keep = d['valid'].any(1)
        if self.points:
            keep &= d['labels'] != self.ignore
            keep = np.flatnonzero(keep)
            if len(keep) > self.points:
                keep = keep[torch.randperm(len(keep))[:self.points].numpy()]
        return {key: torch.from_numpy(d[key][keep]) for key in ('probs', 'valid', 'geometry', 'labels')}


def cat(samples):
    return {key: torch.cat([s[key] for s in samples]) for key in samples[0]}


def evaluate(head, loader, classes, device):
    """Holdout reports of the head and of the plain product (the head's starting point)."""
    head.eval()
    scores = {'head': Confusion(len(classes), classes.ignore), 'product': Confusion(len(classes), classes.ignore)}
    with torch.no_grad():
        for batch in loader:
            probs, valid, geometry = (batch[k].to(device) for k in ('probs', 'valid', 'geometry'))
            labels = batch['labels'].numpy()
            scores['head'].update(head(probs, valid, geometry).argmax(1).cpu().numpy(), labels)
            log_p = probs.float().clamp(1e-6, 1).log() * valid.float()[..., None]
            scores['product'].update(log_p.sum(1).argmax(1).cpu().numpy(), labels)
    return {name: c.report(classes.names) for name, c in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--salsanext', type=Path, required=True, help='SalsaNext checkpoint (best.pt)')
    parser.add_argument('--rfdetr', required=True, help='RF-DETR-Seg checkpoint')
    parser.add_argument('--rfdetr-size', choices=SIZES, default='medium', help='the size the weights were trained as')
    parser.add_argument('--split', choices=('train', 'valid', 'test'), default='valid',
                        help="train on this split of SalsaNext's split.json (next to its checkpoint)")
    parser.add_argument('--recordings', nargs='+', type=Path, help='or: these recording dirs')
    parser.add_argument('--out', type=Path, help='default: <salsanext dir>/fusion_head')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help="its 'fusion' section is used")
    parser.add_argument('--cameras', nargs='+', default=['main', 'surround'], help='the head\'s camera inputs, in order')
    parser.add_argument('--threshold', type=float, default=0.1, help='RF-DETR confidence threshold')
    parser.add_argument('--holdout-frac', type=float, default=0.2)
    parser.add_argument('--layers', type=int, default=3, help='linear layers of the MLP')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--no-geometry', action='store_true', help="don't feed the point's range and height")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32, help='scans per step')
    parser.add_argument('--points-per-scan', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cache-batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.salsanext, map_location='cpu', weights_only=False)
    classes = Classes(checkpoint['classes'])
    fusion_cfg = Classes.load(args.config).config['fusion']
    recordings = args.recordings or load_split(args.salsanext.parent / 'split.json')[args.split]
    out = args.out or args.salsanext.parent / 'fusion_head'
    out.mkdir(parents=True, exist_ok=True)

    entries = build_cache(args, classes, checkpoint, fusion_cfg, recordings, out / 'cache')
    train_files, holdout_files = holdout_split(entries, args.holdout_frac)
    if not holdout_files:
        holdout_files = train_files
    print(f'{len(train_files)} train / {len(holdout_files)} holdout scans')

    train_set = CachedScans(train_files, classes.ignore, args.points_per_scan)
    freq = np.zeros(len(classes), dtype=np.float64)
    for f in tqdm(train_files, desc='class frequencies', leave=False):
        labels = np.load(f)['labels']
        freq += np.bincount(labels[labels != classes.ignore], minlength=len(classes))[:len(classes)]
    weights = np.where(freq > 0, 1.0 / np.sqrt(freq / freq.sum() + 1e-3), 0.0)
    weights = torch.tensor(weights / weights[freq > 0].mean(), dtype=torch.float32, device=args.device)
    print('class weights', dict(zip(classes.names, weights.round(decimals=2).tolist())))

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=cat,
                              drop_last=len(train_set) > args.batch_size)
    holdout_loader = DataLoader(CachedScans(holdout_files, classes.ignore), 8, num_workers=args.workers,
                                collate_fn=cat)
    head = FusionHead(len(classes), args.cameras, args.hidden, args.layers, not args.no_geometry).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_loader),
                                                    pct_start=0.1)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=classes.ignore)

    best, history = -1.0, []
    for epoch in range(args.epochs):
        head.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f'epoch {epoch}', leave=False):
            probs, valid, geometry = (batch[k].to(args.device) for k in ('probs', 'valid', 'geometry'))
            loss = cross_entropy(head(probs, valid, geometry), batch['labels'].long().to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        report = evaluate(head, holdout_loader, classes, args.device)
        history.append({'epoch': epoch, 'loss': total / max(1, len(train_loader)), **report})
        print(f'epoch {epoch}: loss {history[-1]["loss"]:.3f}, holdout mIoU head {report["head"]["miou"]:.3f} '
              f'(product {report["product"]["miou"]:.3f}), person IoU {report["head"]["iou"].get("person")}')
        saved = {'model': head.state_dict(), 'config': head.config, 'classes': classes.config,
                 'salsanext': str(args.salsanext), 'rfdetr': args.rfdetr, 'threshold': args.threshold,
                 'epoch': epoch, 'holdout': report}
        torch.save(saved, out / 'last.pt')
        if report['head']['miou'] is not None and report['head']['miou'] > best:
            best = report['head']['miou']
            torch.save(saved, out / 'best.pt')
        with open(out / 'history.json', 'w') as f:
            json.dump(history, f, indent=1)
    print(f'best holdout mIoU {best:.3f} -> {out / "best.pt"}')


if __name__ == '__main__':
    main()
