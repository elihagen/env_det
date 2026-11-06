#!/usr/bin/env python3
"""Write recordings with their per-point predictions to .mcap files for Foxglove / Lichtblick.

    python scripts/export_mcap.py --split test \
        --pred lidar=outputs/salsanext/pred_test --pred fused=outputs/salsanext/fusion_test

One file per recording (<out>/<run>_<scene>.mcap), open it with "Open local file". Per lidar stamp
(both lidars merged, in base_link):

    /gt, /<name>                    point clouds in class colours (configs/classes.yaml palette;
                                    colour mode "RGBA (separate fields)"), plus a `label` field
    /<name>/errors                  green = matches ground truth, red = wrong, grey = no ground truth
    /camera/<cam>                   the camera frame nearest in time (as used by the fusion)
    /camera/<cam>/<name>, .../gt    the same frame with the scan's points painted by class
    /legend                         class colours (one image at the start)
    /tf                             world -> base_link (recorded robot pose)

--pred NAME=DIR takes the <DIR>/<run>/<scene>/<frame>.npy files of infer_salsanext.py or
infer_fusion.py (one train id per point, 255 = none).
"""
import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from google.protobuf.timestamp_pb2 import Timestamp
from PIL import Image

from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from mcap_protobuf.writer import Writer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from infer_fusion import paint  # noqa: E402
from infer_salsanext import recording_name  # noqa: E402
from lero_seg import geometry  # noqa: E402
from lero_seg.classes import Classes  # noqa: E402
from lero_seg.fusion_dataset import FusionDataset  # noqa: E402
from lero_seg.recordings import load_split  # noqa: E402
from lero_seg.visualize import legend  # noqa: E402

CLOUD_DTYPE = np.dtype({'names': ['x', 'y', 'z', 'red', 'green', 'blue', 'alpha', 'label'],
                        'formats': ['<f4', '<f4', '<f4', 'u1', 'u1', 'u1', 'u1', 'u1'],
                        'offsets': [0, 4, 8, 12, 13, 14, 15, 16], 'itemsize': 20})
F32, U8 = PackedElementField.FLOAT32, PackedElementField.UINT8
CLOUD_FIELDS = [PackedElementField(name=n, offset=CLOUD_DTYPE.fields[n][1], type=F32 if n in 'xyz' else U8)
                for n in CLOUD_DTYPE.names]
ERROR_COLOURS = np.array([[60, 200, 60], [230, 40, 40], [110, 110, 110]], dtype=np.uint8)  # ok, wrong, no gt


def stamp(ns: int) -> Timestamp:
    return Timestamp(seconds=ns // 1_000_000_000, nanos=ns % 1_000_000_000)


def cloud(ns: int, frame: str, xyz: np.ndarray, colours: np.ndarray, labels: np.ndarray) -> PointCloud:
    data = np.zeros(len(xyz), dtype=CLOUD_DTYPE)
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['red'], data['green'], data['blue'] = colours[:, 0], colours[:, 1], colours[:, 2]
    data['alpha'], data['label'] = 255, labels
    return PointCloud(timestamp=stamp(ns), frame_id=frame, pose=Pose(orientation=Quaternion(w=1.0)),
                      point_stride=CLOUD_DTYPE.itemsize, fields=CLOUD_FIELDS, data=data.tobytes())


def jpeg(ns: int, frame: str, rgb: np.ndarray) -> CompressedImage:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='JPEG', quality=85)
    return CompressedImage(timestamp=stamp(ns), frame_id=frame, data=buffer.getvalue(), format='jpeg')


def export(recording: Path, preds: dict[str, Path], dataset: FusionDataset, indices: list[int], classes: Classes,
           out: Path) -> None:
    palette = classes.palette()
    name = recording_name(recording)
    by_stamp = defaultdict(list)
    for i in indices:
        by_stamp[dataset.lidar.items[i]['stamp_ns']].append(i)

    with open(out, 'wb') as f, Writer(f) as writer:
        first = min(by_stamp)
        legend_img = legend(classes.names, palette, height=22 * len(classes) + 10)
        writer.write_message('/legend', jpeg(first, 'legend', legend_img), log_time=first, publish_time=first)
        for ns in sorted(by_stamp):
            xyz, labels, predicted, images = [], [], defaultdict(list), {}
            for i in by_stamp[ns]:
                item = dataset.lidar.items[i]
                sample = dataset[i]
                _, _, points, gt = dataset.lidar.project(i)
                xyz.append(geometry.apply(dataset._base_from_lidar(item), points))
                labels.append(gt)
                scan_preds = {}
                for pred_name, pred_dir in preds.items():
                    path = pred_dir / name / f'{item["frame"]}.npy'
                    if path.is_file():
                        scan_preds[pred_name] = np.load(path)
                        predicted[pred_name].append(scan_preds[pred_name])
                # Paint this scan's points into the camera frames it was fused with (both lidars share them).
                for view in sample['views']:
                    key = view['camera']
                    if key not in images:
                        images[key] = {'raw': view['rgb'], 'gt': view['rgb'].copy(),
                                       **{p: view['rgb'].copy() for p in preds}}
                    seen_gt = view['visible'] & (gt != classes.ignore)
                    images[key]['gt'] = paint(images[key]['gt'], view['uv'], seen_gt, gt.astype(np.uint8), palette)
                    for pred_name, pred in scan_preds.items():
                        images[key][pred_name] = paint(images[key][pred_name], view['uv'], view['visible'], pred,
                                                       palette)

            xyz, labels = np.concatenate(xyz), np.concatenate(labels).astype(np.uint8)
            writer.write_message('/gt', cloud(ns, 'base_link', xyz, palette[labels], labels), log_time=ns, publish_time=ns)
            for pred_name, parts in predicted.items():
                if len(parts) != len(by_stamp[ns]):
                    continue  # a scan of this stamp has no prediction
                pred = np.concatenate(parts).astype(np.uint8)
                writer.write_message(f'/{pred_name}', cloud(ns, 'base_link', xyz, palette[pred], pred),
                                     log_time=ns, publish_time=ns)
                error = np.where(labels == classes.ignore, 2, (pred != labels).astype(np.int64))
                writer.write_message(f'/{pred_name}/errors', cloud(ns, 'base_link', xyz, ERROR_COLOURS[error], pred),
                                     log_time=ns, publish_time=ns)
            for cam, variants in images.items():
                for variant, rgb in variants.items():
                    topic = f'/camera/{cam}' if variant == 'raw' else f'/camera/{cam}/{variant}'
                    writer.write_message(topic, jpeg(ns, cam, rgb), log_time=ns, publish_time=ns)

            pose = dataset.lidar.items[by_stamp[ns][0]]['robot_pose']
            tf = FrameTransform(timestamp=stamp(ns), parent_frame_id='world', child_frame_id='base_link',
                                translation=Vector3(x=pose[0], y=pose[1], z=pose[2]),
                                rotation=Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]))
            writer.write_message('/tf', tf, log_time=ns, publish_time=ns)
    print(f'{out}: {len(by_stamp)} stamps, {len(indices)} scans, predictions: {list(preds) or "none"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=('train', 'valid', 'test'), help='recordings of this split (--split-file)')
    parser.add_argument('--split-file', type=Path, default=ROOT / 'outputs/salsanext/split.json')
    parser.add_argument('--recordings', nargs='+', type=Path)
    parser.add_argument('--pred', action='append', default=[], metavar='NAME=DIR', help='per-point predictions')
    parser.add_argument('--salsanext', type=Path, default=ROOT / 'outputs/salsanext/best.pt',
                        help='only for its classes and normalisation stats (the dataset loader needs them)')
    parser.add_argument('--cameras', nargs='+', default=['main', 'surround'])
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/mcap')
    args = parser.parse_args()
    if not (args.split or args.recordings):
        parser.error('give --split or --recordings')

    preds = {}
    for spec in args.pred:
        pred_name, _, pred_dir = spec.partition('=')
        if not pred_dir:
            parser.error(f'--pred {spec!r}: expected NAME=DIR')
        preds[pred_name] = Path(pred_dir) if Path(pred_dir).is_absolute() else ROOT / pred_dir
    checkpoint = torch.load(args.salsanext, map_location='cpu', weights_only=False)
    classes = Classes(checkpoint['classes'])
    recordings = args.recordings or load_split(args.split_file)[args.split]
    dataset = FusionDataset(recordings, classes, checkpoint['stats'], classes.config['fusion'], args.cameras)
    args.out.mkdir(parents=True, exist_ok=True)
    for recording in recordings:
        indices = [i for i, item in enumerate(dataset.lidar.items) if item['recording'] == recording]
        export(recording, preds, dataset, indices, classes, args.out / f'{recording_name(recording).replace("/", "_")}.mcap')


if __name__ == '__main__':
    main()
