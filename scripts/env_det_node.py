#!/usr/bin/env python3
"""ROS 2 node: semantic segmentation of each lidar scan (SalsaNext, optionally fused with RF-DETR-Seg)
for the navigation stack.

    python scripts/env_det_node.py --config configs/env_det.yaml [--ros-args ...]

Subscribes to the lidar topics (and, with an RF-DETR checkpoint, the cameras) and publishes per scan,
with the scan's own header (stamp + lidar frame), so TF places the points where they were measured:

    <namespace>/points        every point: x y z confidence instance label (lero_seg/ros_cloud.py)
    <namespace>/<group>       only the group's classes above min_confidence, e.g. stop_points
                              (person, animal) and avoid_points (obstacle), for a grid map layer each

Labels are the train ids of configs/classes.yaml. Instance ids are per scan (not tracked).
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lero_seg.env_det import CameraView, EnvDetector, cluster_instances  # noqa: E402
from lero_seg.geometry import USD_TO_OPTICAL, quat_matrix, transform  # noqa: E402
from lero_seg.ros_cloud import make_cloud, read_points  # noqa: E402


def stamp_ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def matrix(tf) -> np.ndarray:
    """4x4 target <- source from a geometry_msgs/TransformStamped."""
    t, q = tf.transform.translation, tf.transform.rotation
    return transform(quat_matrix(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


def image_rgb(msg: Image) -> np.ndarray:
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}
    if msg.encoding not in channels:
        raise ValueError(f'unsupported image encoding {msg.encoding!r}')
    n = channels[msg.encoding]
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    image = rows[:, :msg.width * n].reshape(msg.height, msg.width, n)[..., :3]
    return np.ascontiguousarray(image[..., ::-1] if msg.encoding.startswith('bgr') else image)


class Camera:
    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self.frames: deque = deque(maxlen=10)  # (stamp_ns, Image)
        self.info: CameraInfo | None = None

    @property
    def frame_id(self) -> str | None:
        return self.cfg.get('frame') or (self.info.header.frame_id if self.info else None)

    def nearest(self, t_ns: int, max_dt_ns: float):
        if not self.frames:
            return None
        stamp, msg = min(self.frames, key=lambda f: abs(f[0] - t_ns))
        return (stamp, msg) if abs(stamp - t_ns) <= max_dt_ns else None


class EnvDetNode(Node):
    def __init__(self, cfg: dict):
        super().__init__('env_det', parameter_overrides=[Parameter('use_sim_time', value=cfg.get('use_sim_time', False))])
        self.cfg = cfg
        resolve = lambda p: str(p if Path(p).is_absolute() else ROOT / p)  # noqa: E731
        self.detector = EnvDetector(resolve(cfg['salsanext']), cfg.get('device', 'cuda'),
                                    rfdetr=resolve(cfg['rfdetr']) if cfg.get('rfdetr') else None,
                                    rfdetr_size=cfg.get('rfdetr_size', 'medium'),
                                    rfdetr_threshold=cfg.get('rfdetr_threshold', 0.1),
                                    fusion_head=resolve(cfg['fusion_head']) if cfg.get('fusion_head') else None)
        classes = self.detector.classes
        self.groups = {name: np.array([classes.names.index(c) for c in names], dtype=np.uint8)
                       for name, names in cfg.get('groups', {}).items()}
        self.cluster_ids = sorted({int(c) for ids in self.groups.values() for c in ids})

        self.tf_buffer = Buffer()
        # Own thread: lookups wait (briefly) inside the scan callback, which blocks this node's executor.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        ns = cfg.get('namespace', '/perception/env_det').rstrip('/')
        self.pub_all = self.create_publisher(PointCloud2, f'{ns}/points', 5) if cfg.get('publish_all_points', True) else None
        self.pub_groups = {name: self.create_publisher(PointCloud2, f'{ns}/{name}', 5) for name in self.groups}

        self.cameras = {}
        if self.detector.detector is not None:
            for name, cam_cfg in (cfg.get('cameras') or {}).items():
                camera = self.cameras[name] = Camera(name, cam_cfg)
                self.create_subscription(Image, cam_cfg['image'],
                                         lambda msg, c=camera: c.frames.append((stamp_ns(msg.header.stamp), msg)),
                                         qos_profile_sensor_data)
                self.create_subscription(CameraInfo, cam_cfg['info'], lambda msg, c=camera: setattr(c, 'info', msg),
                                         qos_profile_sensor_data)
        for topic in cfg['lidar_topics']:
            self.create_subscription(PointCloud2, topic, self.on_scan, qos_profile_sensor_data)

        self.timings: deque = deque(maxlen=100)
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            f'lidar {cfg["lidar_topics"]} -> {ns}/points + {[f"{ns}/{g}" for g in self.groups]}; cameras: '
            f'{list(self.cameras) or "none (lidar only)"}')

    def lookup(self, target: str, source: str, stamp, source_stamp=None) -> np.ndarray | None:
        """target <- source; with source_stamp through fixed_frame (target at `stamp`, source at
        `source_stamp`), compensating the robot's motion in between."""
        try:
            if source_stamp is None:
                return matrix(self.tf_buffer.lookup_transform(target, source, stamp, Duration(seconds=0.05)))
            return matrix(self.tf_buffer.lookup_transform_full(target, stamp, source, source_stamp,
                                                               self.cfg.get('fixed_frame', 'odom'),
                                                               Duration(seconds=0.05)))
        except TransformException as ex:
            self.get_logger().warn(f'no transform {source} -> {target}: {ex}', throttle_duration_sec=5.0)
            return None

    def camera_views(self, header) -> list[CameraView]:
        views = []
        t_ns = stamp_ns(header.stamp)
        for camera in self.cameras.values():
            hit = camera.nearest(t_ns, self.cfg.get('max_dt_ms', 30) * 1e6)
            if hit is None or camera.info is None or camera.frame_id is None:
                continue
            image_ns, msg = hit
            cam_from_lidar = self.lookup(camera.frame_id, header.frame_id, Time(nanoseconds=image_ns),
                                         source_stamp=Time.from_msg(header.stamp))
            if cam_from_lidar is None:
                continue
            optical = USD_TO_OPTICAL @ cam_from_lidar if camera.cfg.get('usd_camera', False) else cam_from_lidar
            views.append(CameraView(key=(camera.name, image_ns), rgb=image_rgb(msg),
                                    K=np.asarray(camera.info.k, dtype=np.float64).reshape(3, 3),
                                    width=camera.info.width, height=camera.info.height, optical_from_lidar=optical))
        return views

    def on_scan(self, msg: PointCloud2) -> None:
        start = time.perf_counter()
        points = read_points(msg)
        if len(points) == 0:
            return
        result = self.detector(points, self.camera_views(msg.header) if self.cameras else [])
        labels, confidence = result.labels, result.confidence

        instance = np.zeros(len(points), dtype=np.uint16)
        if self.cluster_ids:
            candidates = np.isin(labels, self.cluster_ids) & (confidence >= self.cfg.get('min_confidence', 0.5))
            if candidates.any():
                to_cluster = self.lookup(self.cfg.get('cluster_frame', 'base_footprint'), msg.header.frame_id,
                                         Time.from_msg(msg.header.stamp))
                xy = (points[:, :3] if to_cluster is None else
                      points[:, :3] @ to_cluster[:3, :3].T + to_cluster[:3, 3])[:, :2]
                instance[candidates] = cluster_instances(
                    xy[candidates], labels[candidates], self.cluster_ids,
                    self.cfg.get('cluster_cell', 0.3), self.cfg.get('cluster_min_points', 3))

        xyz = points[:, :3]
        if self.pub_all is not None:
            self.pub_all.publish(make_cloud(msg.header, xyz, labels, confidence, instance))
        for name, class_ids in self.groups.items():
            keep = np.isin(labels, class_ids) & (confidence >= self.cfg.get('min_confidence', 0.5))
            self.pub_groups[name].publish(make_cloud(msg.header, xyz[keep], labels[keep], confidence[keep],
                                                     instance[keep]))
        self.timings.append((time.perf_counter() - start, len(result.cameras_used)))

    def report(self) -> None:
        if self.timings:
            seconds = np.array([t for t, _ in self.timings])
            cameras = np.mean([c for _, c in self.timings])
            self.get_logger().info(f'{len(seconds)} scans: {1000 * seconds.mean():.1f} ms mean, '
                                   f'{1000 * seconds.max():.1f} ms max, {cameras:.1f} cameras fused per scan')
            self.timings.clear()
        else:
            self.get_logger().warn(f'no scans received on {self.cfg["lidar_topics"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'env_det.yaml')
    args, ros_args = parser.parse_known_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rclpy.init(args=[sys.argv[0], *ros_args])
    node = EnvDetNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
