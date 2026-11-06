"""sensor_msgs/PointCloud2 <-> numpy for the env_det node (needs ROS 2's sensor_msgs).

Output layout of every env_det cloud (one point = 20 bytes):
    x, y, z      float32   in the frame of the input scan (header copied from it)
    confidence   float32   fused probability of `label`
    instance     uint16    per-scan object id, 0 = none (not tracked across scans)
    label        uint8     train id (configs/classes.yaml), 255 = no sensor saw the point
"""
import array

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

_NUMPY = {PointField.INT8: 'i1', PointField.UINT8: 'u1', PointField.INT16: 'i2', PointField.UINT16: 'u2',
          PointField.INT32: 'i4', PointField.UINT32: 'u4', PointField.FLOAT32: 'f4', PointField.FLOAT64: 'f8'}

OUT_FIELDS = [('x', 0, PointField.FLOAT32), ('y', 4, PointField.FLOAT32), ('z', 8, PointField.FLOAT32),
              ('confidence', 12, PointField.FLOAT32), ('instance', 16, PointField.UINT16),
              ('label', 18, PointField.UINT8)]
OUT_DTYPE = np.dtype({'names': [n for n, _, _ in OUT_FIELDS],
                      'formats': [_NUMPY[t] for _, _, t in OUT_FIELDS],
                      'offsets': [o for _, o, _ in OUT_FIELDS], 'itemsize': 20})


def read_points(msg: PointCloud2, names=('x', 'y', 'z', 'intensity')) -> np.ndarray:
    """(N, k) float32 of the named fields that exist (as the datagen recorder does), NaN points dropped."""
    fields = {f.name: f for f in msg.fields}
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.row_step)
    raw = rows[:, :msg.width * msg.point_step].reshape(-1, msg.point_step)
    order = '>' if msg.is_bigendian else '<'
    columns = []
    for name in names:
        if name not in fields:
            continue
        f = fields[name]
        dtype = np.dtype(_NUMPY[f.datatype]).newbyteorder(order)
        columns.append(np.ascontiguousarray(raw[:, f.offset:f.offset + dtype.itemsize]).view(dtype).reshape(-1))
    points = np.stack(columns, axis=1).astype(np.float32)
    return points[np.isfinite(points[:, :3]).all(1)]


def make_cloud(header, xyz: np.ndarray, labels: np.ndarray, confidence: np.ndarray,
               instance: np.ndarray | None = None) -> PointCloud2:
    data = np.zeros(len(xyz), dtype=OUT_DTYPE)
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['confidence'], data['label'] = confidence, labels
    if instance is not None:
        data['instance'] = instance
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name=n, offset=o, datatype=t, count=1) for n, o, t in OUT_FIELDS]
    msg.is_bigendian = False
    msg.point_step = OUT_DTYPE.itemsize
    msg.row_step = msg.point_step * len(xyz)
    msg.is_dense = True
    msg.data = array.array('B')
    msg.data.frombytes(data.tobytes())
    return msg
