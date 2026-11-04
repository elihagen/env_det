#!/usr/bin/env python3
"""Train SalsaNext on the recordings' lidar scans (per-point labels from the simulator).

    python scripts/train_salsanext.py --data /path/to/datagen_runs --out outputs/salsanext

Splits the recordings by scene into train/valid/test (saved as split.json), trains with
weighted cross-entropy + Lovasz-Softmax as in the paper, and keeps the checkpoint with the
best validation mIoU (best.pt; last.pt every epoch).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lero_seg.classes import DEFAULT_CONFIG, Classes  # noqa: E402
from lero_seg.lidar_dataset import RangeImageDataset  # noqa: E402
from lero_seg.metrics import Confusion  # noqa: E402
from lero_seg.recordings import find_recordings, load_split, save_split, split_recordings  # noqa: E402
from lero_seg.salsanext import SalsaNext, lovasz_softmax  # noqa: E402


def evaluate(model, loader, classes, device):
    model.eval()
    confusion = Confusion(len(classes), classes.ignore)
    with torch.no_grad():
        for image, label, mask in loader:
            pred = model(image.to(device)).argmax(1).cpu()
            pred[~mask] = classes.ignore
            confusion.update(pred.numpy(), label.numpy())
    return confusion.report(classes.names)


def init_from(model, state_dict):
    """Copy pretrained weights layer by layer in order (the reference implementation names its
    layers differently, but in the same order), all but the classifier."""
    ours = model.state_dict()
    theirs = list(state_dict.values())
    if len(theirs) != len(ours):
        raise ValueError(f'{len(theirs)} pretrained tensors for {len(ours)} in the model')
    loaded = {name: value for (name, own), value in zip(ours.items(), theirs)
              if not name.startswith('logits.') and own.shape == value.shape}
    model.load_state_dict(loaded, strict=False)
    print(f'initialised {len(loaded)}/{len(ours)} tensors from pretrained weights')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', nargs='+', required=True, help='folders searched for recordings')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--split', type=Path, help='reuse a split.json instead of drawing a new one')
    parser.add_argument('--valid-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01, help='SGD learning rate (paper: 0.01)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--init', type=Path, help='start from these weights (e.g. the SemanticKITTI state_dict.pt) '
                                                  'instead of random ones; the classifier is always new')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    classes = Classes.load(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    split = load_split(args.split) if args.split else split_recordings(
        find_recordings(args.data), args.valid_frac, args.test_frac, args.seed)
    save_split(split, args.out / 'split.json')
    print({k: len(v) for k, v in split.items()}, 'recordings per split')

    lidar_cfg = classes.config['lidar']
    train_set = RangeImageDataset(split['train'], classes, lidar_cfg, augment=True)
    stats = train_set.compute_stats()
    valid_set = RangeImageDataset(split['valid'] or split['train'], classes, lidar_cfg, stats=stats)
    print(f'{len(train_set)} train / {len(valid_set)} valid scans')

    # Inverse square-root class frequency, as rare classes (people!) would otherwise vanish.
    freq = train_set.class_frequencies().astype(np.float64)
    weights = np.where(freq > 0, 1.0 / np.sqrt(freq / freq.sum() + 1e-3), 0.0)
    weights = torch.tensor(weights / weights[freq > 0].mean(), dtype=torch.float32, device=args.device)
    print('class weights', dict(zip(classes.names, weights.round(decimals=2).tolist())))

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    valid_loader = DataLoader(valid_set, args.batch_size, num_workers=args.workers)
    model = SalsaNext(len(classes))
    if args.init:
        init_from(model, torch.load(args.init, map_location='cpu', weights_only=True))
    model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps, pct_start=0.05)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=classes.ignore)
    scaler = torch.amp.GradScaler(enabled=args.device.startswith('cuda'))

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for image, label, _ in tqdm(train_loader, desc=f'epoch {epoch}', leave=False):
            image, label = image.to(args.device), label.to(args.device)
            with torch.autocast(args.device.split(':')[0], enabled=args.device.startswith('cuda')):
                logits = model(image)
            logits = logits.float()
            loss = cross_entropy(logits, label) + lovasz_softmax(logits.softmax(1), label, classes.ignore)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
        report = evaluate(model, valid_loader, classes, args.device)
        history.append({'epoch': epoch, 'loss': total / max(1, len(train_loader)), **report})
        print(f'epoch {epoch}: loss {history[-1]["loss"]:.3f}, valid mIoU {report["miou"]:.3f}, '
              f'person IoU {report["iou"].get("person")}')
        checkpoint = {'model': model.state_dict(), 'classes': classes.config, 'stats': stats,
                      'epoch': epoch, 'valid': report}
        torch.save(checkpoint, args.out / 'last.pt')
        if report['miou'] is not None and report['miou'] > best:
            best = report['miou']
            torch.save(checkpoint, args.out / 'best.pt')
        with open(args.out / 'history.json', 'w') as f:
            json.dump(history, f, indent=1)
    print(f'best valid mIoU {best:.3f} -> {args.out / "best.pt"}')


if __name__ == '__main__':
    main()
