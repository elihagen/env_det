# lero_sim_segmentation

Semantic segmentation of the LERO Isaac Sim datagen recordings
(`lero_isaac_stack/scripts/datagen/run_datagen.sh`):

- **Lidar: [SalsaNext](https://arxiv.org/abs/2003.03653)**, one SICK multiScan165 scan at a time, projected
  onto a 16 x 1024 range image (16 laser layers x 360 deg). Per-point labels come straight from the
  simulator (`point_labels/`).
- **Camera: [RF-DETR-Seg](https://github.com/roboflow/rf-detr)**, fine-tuned from COCO. RF-DETR is an
  instance-segmentation model, so the recordings are exported as COCO instances (each person, obstacle,
  tree, bush separately; ground, grass, sky, crops as one region per class) and its masks are painted
  back into a per-pixel class map for scoring.

Both use the same classes (`configs/classes.yaml`): ground, grass, crop, crop_row, person, animal,
obstacle, tree, bush, sky, robot. The recorder's own labels are mapped onto them there.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
```

Or in Docker (needs the NVIDIA container toolkit):

```bash
docker compose build
docker compose run --rm seg python scripts/train_salsanext.py --data /mnt/nas/eh_datasets/isaac --out outputs/salsanext
```

The repo is mounted into the container (edits and `outputs/` stay on the host) and `/mnt/nas` is
mounted read-only at the same path, since `split.json`/`sources.json` hold absolute recording paths.

VS Code (also over Remote-SSH): the Run and Debug panel has a configuration per script
(`.vscode/launch.json`), incl. 1-epoch smoke tests. Each starts its script in the container under
debugpy (`scripts/debug_listen.py`, port 5678 on localhost) and attaches to it.

The GPU is shared with Isaac Sim on the datagen VM (it holds ~16 GB of the 24 GB): stop the
`isaac_sim` container while training, or use small batches.

## Data

Point `--data` at one or more folders holding recordings, e.g. a run directory
(`.nr_sim/datagen/run_*`) or the NAS archive. Every `recording/` below it is one scene. Scenes are split
into train/valid/test (default 80/10/10, `split.json`), never frames of one scene across splits.
Reuse one split for both models with `--split`.

Used per recording: `points/`, `point_labels/`, `lidar_index.jsonl`, `classes.json` (lidar) and
`rgb/`, `semantic/`, `camera_index.jsonl`, plus `instance/` + `persons.jsonl` where recorded
(camera; without them all objects of a class become one instance). Both cameras are used if present.

## Lidar: SalsaNext

```bash
python scripts/train_salsanext.py --data /path/to/runs --out outputs/salsanext
python scripts/infer_salsanext.py --checkpoint outputs/salsanext/best.pt --split test
python scripts/infer_salsanext.py --checkpoint outputs/salsanext/best.pt --recordings NEW_SCENE/recording \
    --out outputs/salsanext/pred_new --ply
```

Inference writes `preview/` images (range image and bird's-eye view, ground truth | prediction) and with
`--ply` one point cloud per scan coloured by class (open in CloudCompare or MeshLab). Recordings without
`point_labels/` are predicted and previewed but not scored.

`--init checkpoints/salsanext_semantickitti/state_dict.pt` starts from the authors' SemanticKITTI
model ([pretrained.zip](https://drive.google.com/file/d/1utfzooTDAlV5M6XGvCE0-L-vbdLe_2rD/view), MIT;
64-beam Velodyne, 20 classes) instead of random weights; only the classifier is new. `state_dict.pt`
holds just its weights, re-saved from the zip's `pretrained/SalsaNext`.

Training follows the paper: weighted cross-entropy + Lovasz-Softmax, SGD (lr 0.01, momentum 0.9),
random azimuth rotation and left/right mirroring. The Bayesian uncertainty part of SalsaNext is left
out. Each projected point gets its pixel's class; points matching no layer get 255.

## Camera: RF-DETR-Seg

```bash
python scripts/export_coco.py --data /path/to/runs --out datasets/coco_seg --split outputs/salsanext/split.json
python scripts/train_rfdetr.py --dataset datasets/coco_seg --out outputs/rfdetr --size medium
python scripts/infer_rfdetr.py --weights outputs/rfdetr/<checkpoint>.pth --dataset datasets/coco_seg --split test
```

The export symlinks images into the dataset (pass `--copy` to move it to another machine) and keeps
every 2nd frame (`--every-n`). RF-DETR's own validation reports mask mAP per class during training;
`infer_rfdetr.py` reports per-pixel IoU and writes previews (rgb | prediction | ground truth).

## Late fusion: SalsaNext + RF-DETR-Seg

```bash
python scripts/infer_fusion.py --salsanext outputs/salsanext/best.pt --rfdetr outputs/rfdetr/<checkpoint>.pth --split test
```

One dataloader (`lero_seg/fusion_dataset.py`) yields each lidar scan together with the camera frames
nearest in time (`fusion.max_dt_ms`). The range image goes through SalsaNext, the frames through
RF-DETR, and every lidar point gets one fused class:

- **Geometry**: points go lidar → `base_link` (URDF) → world (robot pose at the scan) → `base_link`
  (robot pose at the camera frame) → camera (URDF mount from `fusion.cameras`, `camera_info.json`), so
  the robot's motion between scan and frame is compensated. Points hidden behind nearer ones from the
  camera's viewpoint (`fusion.occlusion_*`) don't take camera evidence.
- **Lidar probabilities**: SalsaNext's softmax at the point's range-image pixel.
- **Camera probabilities**: per class, the highest confidence of that class's masks covering the
  point's pixel; the remaining `1 - best` is spread over all classes, so pixels no mask covers carry
  no camera evidence. Use a low `--threshold` (default 0.1), since confidences are fused rather than cut.
- **Fusion** (`fusion.method`): `product` (p ∝ p_lidar^w_lidar · p_camera^w_camera) or `mean`
  (weighted average), weights global or per class. Outside the camera view a point keeps SalsaNext's
  class.

`metrics.json` scores lidar-only, camera-only and fused predictions per point, over all points and
over the points a camera sees: compare them on the valid split to tune the weights.

### Trained fusion head

Instead of the fixed rule, a small fusion head can learn how to merge the two models, trained on the
lidar points' semantic labels (SalsaNext and RF-DETR stay frozen):

```bash
python scripts/train_fusion.py --salsanext outputs/salsanext/best.pt --rfdetr outputs/rfdetr/<checkpoint>.pth --split valid
python scripts/infer_fusion.py --salsanext outputs/salsanext/best.pt --rfdetr outputs/rfdetr/<checkpoint>.pth \
    --head outputs/salsanext/fusion_head/best.pt --split test
```

- **Model** (`FusionHead`, `lero_seg/fusion.py`): per point, an MLP of 3 linear layers (`--layers`,
  `--hidden` 64) over the log-probabilities and validity of the three sources (lidar, `main`,
  `surround` camera; `--cameras`) plus the point's range and height. Its output is added to the
  sources' summed log-probabilities, and its last layer starts at zero, so training starts from the
  `product` rule and learns the correction (per-class trust, calibration, which sensor to believe when).
- **Data**: both models run once over the split and their per-point outputs are cached in
  `<out>/cache/` (reused while models and settings match), so the epochs take seconds. Train on scans
  the models were not trained on (default: the `valid` split), else the head learns to trust their
  overfitted outputs. The last 20% of each recording (`--holdout-frac`) picks the best epoch;
  `history.json` compares its mIoU with the plain product's.
- **Use**: `infer_fusion.py --head`, and `fusion_head:` in `configs/env_det.yaml` for the ROS node.

## Viewing predictions in Foxglove / Lichtblick

```bash
python scripts/export_mcap.py --split test \
    --pred lidar=outputs/salsanext/pred_test --pred fused=outputs/salsanext/fusion_test
```

This writes one `.mcap` per recording to `outputs/mcap/`; open it with *Open local file*. It
doesn't need ROS, a running sim or a GPU. Per lidar stamp, with both lidars merged in `base_link`:

| Topic | Content |
|---|---|
| `/gt`, `/lidar`, `/fused` | point clouds in class colours |
| `/<name>/errors` | green = right, red = wrong, grey = no ground truth |
| `/camera/main`, `.../gt`, `.../fused` | the camera frame used for fusion, raw and with the points painted by class |
| `/legend` | the class colours |
| `/tf` | `world → base_link` |

`--pred NAME=DIR` takes any output folder of `infer_salsanext.py` or `infer_fusion.py`.

## On the robot: ROS 2 node for the navigation stack

`scripts/env_det_node.py` (ROS 2 Jazzy, config `configs/env_det.yaml`) segments every scan of the
twin's lidars (`/sensors/right_lidar_cloud`, `/sensors/left_lidar_cloud`), each on its own as in
training, with SalsaNext. With `rfdetr:` set, it fuses the cameras' frames nearest in time. The scan
goes to `lero_seg/env_det.py`, which uses the same projection, normalisation and fusion as the
scripts above. Robot poses come from TF: lidar → `fixed_frame` at the scan's stamp → camera at the
frame's stamp. The node publishes with the scan's own header (stamp + lidar frame):

| Topic | Content |
|---|---|
| `/perception/env_det/points` | every point: `x y z confidence instance label` (`lero_seg/ros_cloud.py`) |
| `/perception/env_det/stop_points` | person, animal with confidence ≥ `min_confidence` |
| `/perception/env_det/avoid_points` | obstacle, likewise |

- `label` is the train id from `configs/classes.yaml`; 255 means no sensor saw the point.
- `instance` is a per-scan object id: connected cells of a bird's-eye grid in `base_footprint`. Ids
  are not yet tracked across scans.
- Empty group clouds are published too, so the grid map can tell an object has gone.

```bash
docker compose build ros    # ROS 2 Jazzy + torch (Dockerfile.ros), not part of a plain `docker compose build`
docker compose run --rm ros                                              # the node
docker compose run --rm ros python scripts/fake_obstacle_pub.py --class person --x 4 --visible-sec 10
```

The container uses the same DDS setup as `lero_isaac_stack` (host network, `ROS_DOMAIN_ID` default
0, FastDDS over UDPv4 only). `fake_obstacle_pub.py` publishes a person or obstacle with the node's
cloud layout, for testing the navigation side without the model.

**Grid map layer** (`ros/`):
- `env_det_grid_mapper.launch.py` starts one `outdoor_navigation` `grid_mapper` per group cloud,
  configured by `env_det_grid_mapper.yaml`. Each instance publishes on
  `/perception/env_det/<group>_grid_map`.
- Its `obstacle` layer is non-NaN where an object of the group's classes is, and the cell clears
  `decay_time` (1 s) after the last detection.
- It needs the `grid_mapper` package, e.g. in `lero_isaac_stack`'s devcontainer, where it is built.
  That container doesn't mount this repo, so copy `ros/` into its workspace or mount it:
  `ros2 launch <path>/ros/env_det_grid_mapper.launch.py`.
- `mbf_tag_nav_gridmap_layer.yaml` is the same idea written as layers for an orchvine-style
  `mbf-config.yaml`. It's a proposal for the nav stack and is unverified: that plugin's source
  wasn't available.

## Notes

- Synthetic data only: expect a gap to the real robot; score on real labelled data before trusting it.
- Recordings made before the per-person instance ids (`instance/`) and the waypoint-spawn fix
  (`summary.json` → `start_vs_plan`) can still be used for segmentation, but their scenes don't
  match their `scene_meta.json`.
- `configs/classes.yaml` holds the SICK layer elevations; scans from another lidar need their own.
# env_det
