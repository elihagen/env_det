"""One sample per lidar scan, with what both models need: SalsaNext's range image and the RGB frames
of the cameras nearest in time, plus where each of the scan's points lands in those frames."""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import geometry
from .camera import rgb_path
from .lidar_dataset import RangeImageDataset, normalize
from .recordings import camera_dirs, read_jsonl


class FusionDataset(Dataset):
    def __init__(self, recordings, classes, stats, fusion_cfg, cameras=('main', 'surround')):
        self.lidar = RangeImageDataset(recordings, classes, classes.config['lidar'], stats=stats)
        self.cfg = fusion_cfg
        self.cameras = {rec: self._load_cameras(rec, cameras) for rec in recordings}
        self._lidar_mounts: dict[tuple, np.ndarray] = {}

    def _load_cameras(self, recording, names) -> dict:
        urdf = recording / 'robot_description.urdf'
        out = {}
        for name, cam_dir in camera_dirs(recording).items():
            intrinsics = geometry.camera_intrinsics(cam_dir)
            if name not in names or intrinsics is None or name not in self.cfg['cameras']:
                continue
            base_to_cam = np.linalg.inv(geometry.urdf_transform(urdf, 'base_link', self.cfg['cameras'][name]))
            frames = read_jsonl(cam_dir / 'camera_index.jsonl')
            out[name] = {'dir': cam_dir, 'frames': frames, 'stamps': np.array([f['stamp_ns'] for f in frames]),
                         'optical_from_base': geometry.USD_TO_OPTICAL @ base_to_cam, **intrinsics}
        return out

    def _base_from_lidar(self, item) -> np.ndarray:
        key = (item['recording'], item['frame_id'])
        if key not in self._lidar_mounts:
            self._lidar_mounts[key] = geometry.urdf_transform(item['recording'] / 'robot_description.urdf',
                                                              'base_link', item['frame_id'])
        return self._lidar_mounts[key]

    def __len__(self):
        return len(self.lidar)

    def __getitem__(self, idx):
        proj, _, points, labels = self.lidar.project(idx)
        item = self.lidar.items[idx]
        world_from_lidar = geometry.pose_matrix(item['robot_pose']) @ self._base_from_lidar(item)
        views = []
        for name, cam in self.cameras[item['recording']].items():
            nearest = int(np.abs(cam['stamps'] - item['stamp_ns']).argmin())
            if abs(cam['stamps'][nearest] - item['stamp_ns']) > self.cfg['max_dt_ms'] * 1e6:
                continue
            entry = cam['frames'][nearest]
            to_optical = cam['optical_from_base'] @ np.linalg.inv(geometry.pose_matrix(entry['robot_pose'])) \
                @ world_from_lidar
            uv, visible = geometry.project_to_image(
                geometry.apply(to_optical, points), cam['K'], cam['width'], cam['height'],
                occlusion_cell=self.cfg.get('occlusion_cell', 8), occlusion_tol=self.cfg.get('occlusion_tol', (0.05, 0.3)))
            views.append({'camera': name, 'camera_dir': str(cam['dir']), 'frame': entry['frame'], 'uv': uv,
                          'visible': visible, 'rgb': np.array(Image.open(rgb_path(cam['dir'], entry)).convert('RGB'))})
        return {'range_image': torch.from_numpy(normalize(proj.image, proj.mask, self.lidar.stats)),
                'row': proj.row, 'col': proj.col, 'points': points, 'labels': labels, 'views': views,
                'recording': str(item['recording']), 'frame': item['frame']}


def collate(samples: list[dict]) -> dict:
    """Range images stacked into one batch; per-point arrays and camera views stay per sample."""
    out = {key: [s[key] for s in samples] for key in samples[0]}
    out['range_image'] = torch.stack(out['range_image'])
    return out
