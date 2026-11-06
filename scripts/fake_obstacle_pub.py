#!/usr/bin/env python3
"""Publish a fake detected object as env_det would, to test the navigation side without the model.

    python scripts/fake_obstacle_pub.py --class person --x 4 --y 0            # 4 m ahead, forever
    python scripts/fake_obstacle_pub.py --class obstacle --x 3 --visible-sec 5  # vanishes after 5 s

Publishes the object's surface points with env_det's cloud layout (lero_seg/ros_cloud.py) on its
group topic (person/animal -> <ns>/stop_points, obstacle -> <ns>/avoid_points) and on <ns>/points.
With --visible-sec the object stops being published after that time, but empty clouds keep coming,
like env_det when the object has left: the grid map cell should clear after its decay time.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lero_seg.classes import Classes  # noqa: E402
from lero_seg.ros_cloud import make_cloud  # noqa: E402

# class -> (group topic, shape, size [m])
OBJECTS = {'person': ('stop_points', 'cylinder', (0.25, 1.75)), 'animal': ('stop_points', 'box', (0.8, 0.3, 0.6)),
           'obstacle': ('avoid_points', 'box', (0.5, 0.5, 0.5))}


def surface(shape: str, size, spacing: float = 0.05) -> np.ndarray:
    """Points on the object's surface, standing on z = 0, centred at x = y = 0."""
    if shape == 'cylinder':
        radius, height = size
        angles = np.arange(0, 2 * np.pi, spacing / radius)
        heights = np.arange(0, height, spacing)
        a, h = np.meshgrid(angles, heights)
        return np.stack([radius * np.cos(a), radius * np.sin(a), h], -1).reshape(-1, 3)
    lx, ly, lz = size
    grid = lambda n: np.arange(-n / 2, n / 2 + 1e-9, spacing)  # noqa: E731
    xs, ys, zs = grid(lx), grid(ly), grid(lz) + lz / 2
    faces = []
    for x in (xs[0], xs[-1]):
        y, z = np.meshgrid(ys, zs)
        faces.append(np.stack([np.full(y.size, x), y.ravel(), z.ravel()], -1))
    for y in (ys[0], ys[-1]):
        x, z = np.meshgrid(xs, zs)
        faces.append(np.stack([x.ravel(), np.full(x.size, y), z.ravel()], -1))
    x, y = np.meshgrid(xs, ys)
    faces.append(np.stack([x.ravel(), y.ravel(), np.full(x.size, zs[-1])], -1))
    return np.concatenate(faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--class', dest='cls', choices=OBJECTS, default='person')
    parser.add_argument('--x', type=float, default=4.0, help='position in --frame [m]')
    parser.add_argument('--y', type=float, default=0.0)
    parser.add_argument('--z', type=float, default=0.0, help='height of the object\'s base in --frame [m]')
    parser.add_argument('--frame', default='base_footprint')
    parser.add_argument('--rate', type=float, default=10.0, help='clouds per second [Hz]')
    parser.add_argument('--visible-sec', type=float, help='stop publishing the object after this time')
    parser.add_argument('--namespace', default='/perception/env_det')
    parser.add_argument('--confidence', type=float, default=0.9)
    parser.add_argument('--sim-time', action='store_true', help='stamp with /clock (use_sim_time)')
    args, ros_args = parser.parse_known_args()

    classes = Classes.load()
    group, shape, size = OBJECTS[args.cls]
    xyz = (surface(shape, size) + [args.x, args.y, args.z]).astype(np.float32)
    labels = np.full(len(xyz), classes.names.index(args.cls), dtype=np.uint8)
    confidence = np.full(len(xyz), args.confidence, dtype=np.float32)
    instance = np.ones(len(xyz), dtype=np.uint16)

    rclpy.init(args=[sys.argv[0], *ros_args])
    node = Node('fake_obstacle_pub', parameter_overrides=[Parameter('use_sim_time', value=args.sim_time)])
    ns = args.namespace.rstrip('/')
    pubs = [node.create_publisher(PointCloud2, f'{ns}/{group}', 5), node.create_publisher(PointCloud2, f'{ns}/points', 5)]
    start = node.get_clock().now()
    state = {'visible': None}

    def tick():
        now = node.get_clock().now()
        visible = args.visible_sec is None or (now - start).nanoseconds * 1e-9 < args.visible_sec
        if visible != state['visible']:
            state['visible'] = visible
            node.get_logger().info(f'{args.cls} at ({args.x}, {args.y}) in {args.frame} on {ns}/{group}: '
                                   f'{"visible" if visible else "gone"}')
        header = Header(stamp=now.to_msg(), frame_id=args.frame)
        n = len(xyz) if visible else 0
        cloud = make_cloud(header, xyz[:n], labels[:n], confidence[:n], instance[:n])
        for pub in pubs:
            pub.publish(cloud)

    node.create_timer(1.0 / args.rate, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
