#!/usr/bin/env python3
"""Fine-tune RF-DETR-Seg (COCO-pretrained) on a dataset written by export_coco.py.

    python scripts/train_rfdetr.py --dataset datasets/coco_seg --out outputs/rfdetr --size medium

RF-DETR keeps its best checkpoints in --out (see its log for the file names); pass one to
infer_rfdetr.py --weights.
"""
import argparse

import rfdetr

SIZES = {'nano': 'RFDETRSegNano', 'small': 'RFDETRSegSmall', 'medium': 'RFDETRSegMedium',
         'large': 'RFDETRSegLarge', 'xlarge': 'RFDETRSegXLarge', '2xlarge': 'RFDETRSeg2XLarge'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', choices=SIZES, default='medium')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', default='auto', help="'auto' (CUDA only) or an integer")
    parser.add_argument('--grad-accum-steps', type=int, help='with an integer --batch-size, to reach a larger effective batch')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resolution', type=int, help="input size (multiple of the model's patch size); default: the model's")
    parser.add_argument('--weights', help='start from this checkpoint instead of the COCO-pretrained one')
    args = parser.parse_args()

    model_kwargs = {'pretrain_weights': args.weights} if args.weights else {}
    if args.resolution:
        model_kwargs['resolution'] = args.resolution
    model = getattr(rfdetr, SIZES[args.size])(**model_kwargs)
    train_kwargs = {'dataset_dir': args.dataset, 'epochs': args.epochs, 'lr': args.lr, 'output_dir': args.out,
                    'batch_size': args.batch_size if args.batch_size == 'auto' else int(args.batch_size)}
    if args.grad_accum_steps:
        train_kwargs['grad_accum_steps'] = args.grad_accum_steps
    model.train(**train_kwargs)


if __name__ == '__main__':
    main()
