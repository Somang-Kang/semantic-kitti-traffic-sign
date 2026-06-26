import argparse
import os
from collections import deque, defaultdict

import numpy as np
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN
from mmdet3d.apis import LidarSeg3DInferencer


DEFAULT_DATA_ROOT = "./data/semantickitti/sequences"
DEFAULT_LABEL_ROOT = "./data/semantickitti/sequences"

DEFAULT_CONFIG = "./configs/cylinder3d/cylinder3d_4xb4-3x_semantickitti.py"
DEFAULT_CKPT = "./checkpoints/cylinder3d_4xb4_3x_semantickitti_20230318_191107-822a8c31.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SemanticKITTI traffic sign GT / prediction visualizer"
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequence",type=str,default="08",help="SemanticKITTI sequence (e.g. 00, 08)")
    parser.add_argument("--label-root", type=str, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["only-gt", "only-pred", "both"],
    )

    parser.add_argument("--gt-class", type=int, default=81)
    parser.add_argument("--model-class", type=int, default=18)

    parser.add_argument(
        "--filtering",
        action="store_true",
        help="enable distance filtering and score filtering",
    )
    parser.add_argument(
        "--dist-filtering",
        action="store_true",
        help="enable distance filtering",
    )

    parser.add_argument(
        "--score-filtering",
        action="store_true",
        help="enable score filtering",
    )

    parser.add_argument("--max-distance", type=float, default=65.0)
    parser.add_argument("--score-thr", type=float, default=0.7)

    parser.add_argument(
        "--cluster",
        type=str,
        default="dbscan",
        choices=["dbscan", "vcc", "none"],
    )
    parser.add_argument("--db-eps", type=float, default=1.0)
    parser.add_argument("--db-min-samples", type=int, default=3)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--max-extent-x", type=float, default=5.0)
    parser.add_argument("--max-extent-y", type=float, default=5.0)
    parser.add_argument("--max-extent-z", type=float, default=8.0)

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)

    return parser.parse_args()


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_label(path):
    raw = np.fromfile(path, dtype=np.uint32)
    return raw & 0xFFFF


def _extract_tensor_from_pointdata(obj):
    if torch.is_tensor(obj):
        return obj

    if hasattr(obj, "keys"):
        keys = list(obj.keys())

        for key in [
            "pts_seg_logits",
            "seg_logits",
            "logits",
            "pts_semantic_mask",
            "semantic_mask",
        ]:
            if key in keys:
                value = getattr(obj, key)
                if torch.is_tensor(value):
                    return value

        for key in keys:
            value = getattr(obj, key)
            if torch.is_tensor(value):
                return value

    raise TypeError(f"Cannot extract Tensor from object type: {type(obj)}")


def run_inference(inferencer, points):
    with torch.no_grad():
        result = inferencer(
            dict(points=points),
            show=False,
            return_datasamples=True,
        )

    pred = result["predictions"][0]

    if hasattr(pred, "pts_seg_logits"):
        logits = _extract_tensor_from_pointdata(pred.pts_seg_logits)
        num_points = points.shape[0]

        if logits.ndim != 2:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        if logits.shape[0] == num_points:
            probs = torch.softmax(logits, dim=1)
            pred_scores, pred_labels = probs.max(dim=1)

        elif logits.shape[1] == num_points:
            probs = torch.softmax(logits, dim=0)
            pred_scores, pred_labels = probs.max(dim=0)

        else:
            raise ValueError(
                f"Logits shape {logits.shape} does not match num_points={num_points}"
            )

        return (
            pred_labels.detach().cpu().numpy(),
            pred_scores.detach().cpu().numpy(),
        )

    if hasattr(pred, "pred_pts_seg"):
        pred_mask = _extract_tensor_from_pointdata(pred.pred_pts_seg)
        pred_labels = pred_mask.detach().cpu().numpy()
        pred_scores = np.ones_like(pred_labels, dtype=np.float32)
        return pred_labels, pred_scores

    raise KeyError(
        "Cannot find pts_seg_logits or pred_pts_seg in inferencer output."
    )


def get_distance_mask(points, args):
    if not (args.filtering or args.dist_filtering):
        return np.ones(len(points), dtype=bool)

    dist = np.linalg.norm(points[:, :2], axis=1)
    return dist <= args.max_distance


def get_pred_masks(pred_labels, pred_scores, args):
    pred_all = pred_labels == args.model_class

    if args.filtering or args.score_filtering:
        pred_keep = pred_all & (pred_scores >= args.score_thr)
        pred_low = pred_all & (pred_scores < args.score_thr)
    else:
        pred_keep = pred_all
        pred_low = np.zeros_like(pred_all, dtype=bool)

    return pred_all, pred_keep, pred_low


def is_valid_extent(extent, args):
    return (
        extent[0] <= args.max_extent_x
        and extent[1] <= args.max_extent_y
        and extent[2] <= args.max_extent_z
    )


def make_bbox_from_cluster(cluster, color=(0.0, 0.4, 1.0)):
    min_p = cluster.min(axis=0)
    max_p = cluster.max(axis=0)

    extent = max_p - min_p
    extent = np.maximum(extent, 1e-3)

    box = o3d.geometry.OrientedBoundingBox(
        center=(min_p + max_p) / 2,
        R=np.eye(3),
        extent=extent,
    )
    box.color = color

    return box, extent


def voxel_connected_components(points, voxel_size):
    voxel_map = defaultdict(list)

    for i, p in enumerate(points):
        voxel = tuple(np.floor(p[:3] / voxel_size).astype(np.int32))
        voxel_map[voxel].append(i)

    visited = set()
    clusters = []

    directions = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for voxel in list(voxel_map.keys()):
        if voxel in visited:
            continue

        queue = deque([voxel])
        visited.add(voxel)
        cluster_indices = []

        while queue:
            v = queue.popleft()
            cluster_indices.extend(voxel_map[v])

            vx, vy, vz = v

            for dx, dy, dz in directions:
                nv = (vx + dx, vy + dy, vz + dz)

                if nv in voxel_map and nv not in visited:
                    visited.add(nv)
                    queue.append(nv)

        clusters.append(cluster_indices)

    return clusters


def get_bboxes_from_mask(points, mask, args, color):
    pts = points[mask][:, :3]

    if len(pts) == 0 or args.cluster == "none":
        return []

    if args.cluster == "dbscan":
        if len(pts) < args.db_min_samples:
            return []

        clustering = DBSCAN(
            eps=args.db_eps,
            min_samples=args.db_min_samples,
        ).fit(pts)

        clusters = [
            pts[clustering.labels_ == cid]
            for cid in np.unique(clustering.labels_)
            if cid != -1
        ]

    elif args.cluster == "vcc":
        cluster_indices = voxel_connected_components(
            pts,
            voxel_size=args.voxel_size,
        )
        clusters = [pts[idx] for idx in cluster_indices]

    else:
        raise ValueError(f"Unknown cluster method: {args.cluster}")

    boxes = []

    for cluster in clusters:
        if len(cluster) < args.min_cluster_points:
            continue

        box, extent = make_bbox_from_cluster(cluster, color=color)

        if not is_valid_extent(extent, args):
            continue

        boxes.append(box)

    return boxes


def make_geometry(points, pred_labels, pred_scores, gt_labels, args):
    geometries = []

    xyz = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    intensity = points[:, 3]
    intensity = (intensity - intensity.min()) / (
        intensity.max() - intensity.min() + 1e-6
    )
    intensity = np.power(intensity, 0.6)

    gray = 0.75 + 0.25 * intensity
    colors = np.stack([gray, gray, gray], axis=1)

    distance_mask = get_distance_mask(points, args)

    gt_mask = None
    pred_all = None
    pred_keep = None
    pred_low = None

    if args.mode in ["only-gt", "both"]:
        if gt_labels is None:
            raise ValueError("--mode only-gt/both requires label files.")

        gt_mask = (gt_labels == args.gt_class) & distance_mask

    if args.mode in ["only-pred", "both"]:
        if pred_labels is None or pred_scores is None:
            raise ValueError("--mode only-pred/both requires prediction.")

        pred_all, pred_keep, pred_low = get_pred_masks(
            pred_labels,
            pred_scores,
            args,
        )

        pred_all = pred_all & distance_mask
        pred_keep = pred_keep & distance_mask
        pred_low = pred_low & distance_mask

    if args.mode == "only-gt":
        colors[gt_mask] = [0.0, 1.0, 0.0]

    elif args.mode == "only-pred":
        colors[pred_keep] = [1.0, 0.0, 0.0]

    elif args.mode == "both":
        colors[gt_mask & (~pred_keep)] = [0.0, 1.0, 0.0]
        colors[pred_keep & (~gt_mask)] = [1.0, 0.0, 0.0]

        colors[gt_mask & pred_keep] = [1.0, 1.0, 0.0]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)

    if args.cluster != "none":
        if args.mode in ["only-gt", "both"]:
            geometries.extend(
                get_bboxes_from_mask(
                    points=points,
                    mask=gt_mask,
                    args=args,
                    color=(0.0, 1.0, 0.0),
                )
            )

        if args.mode in ["only-pred", "both"]:
            geometries.extend(
                get_bboxes_from_mask(
                    points=points,
                    mask=pred_keep,
                    args=args,
                    color=(1.0, 0.0, 0.0),
                )
            )

    return geometries


class App:
    def __init__(self, args):
        self.args = args

        args.data_root = os.path.join(
            args.data_root,
            args.sequence,
            "velodyne",
        )

        args.label_root = os.path.join(
            args.data_root,
            args.sequence,
            "labels",
        )

        if not os.path.isdir(args.data_root):
            raise FileNotFoundError(f"DATA_ROOT not found: {args.data_root}")

        if args.mode in ["only-gt", "both"] and not os.path.isdir(args.label_root):
            raise FileNotFoundError(f"LABEL_ROOT not found: {args.label_root}")

        self.files = sorted(
            [f for f in os.listdir(args.data_root) if f.endswith(".bin")]
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No .bin files found in {args.data_root}")

        self.idx = max(0, min(args.start_idx, len(self.files) - 1))

        self.inferencer = None
        if args.mode in ["only-pred", "both"]:
            self.inferencer = LidarSeg3DInferencer(
                model=args.config,
                weights=args.checkpoint,
                device=args.device,
            )

        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    def load_current_frame(self):
        fname = self.files[self.idx]

        lidar_path = os.path.join(self.args.data_root, fname)
        points = load_bin(lidar_path)

        gt_labels = None
        if self.args.mode in ["only-gt", "both"]:
            label_path = os.path.join(
                self.args.label_root,
                fname.replace(".bin", ".label"),
            )
            gt_labels = load_label(label_path)

        pred_labels = None
        pred_scores = None
        if self.args.mode in ["only-pred", "both"]:
            pred_labels, pred_scores = run_inference(
                self.inferencer,
                points,
            )

        return fname, points, gt_labels, pred_labels, pred_scores

    def render_frame(self):
        fname, points, gt_labels, pred_labels, pred_scores = self.load_current_frame()

        geometries = make_geometry(
            points=points,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_labels=gt_labels,
            args=self.args,
        )

        self.vis.clear_geometries()

        for g in geometries:
            self.vis.add_geometry(g)

        distance_mask = get_distance_mask(points, self.args)

        gt_count = -1
        if gt_labels is not None:
            gt_count = int(((gt_labels == self.args.gt_class) & distance_mask).sum())

        if pred_labels is not None:
            pred_all, pred_keep, pred_low = get_pred_masks(
                pred_labels,
                pred_scores,
                self.args,
            )

            pred_all = pred_all & distance_mask
            pred_keep = pred_keep & distance_mask
            pred_low = pred_low & distance_mask

            pred_all_count = int(pred_all.sum())
            pred_keep_count = int(pred_keep.sum())
            pred_low_count = int(pred_low.sum())
        else:
            pred_all_count = -1
            pred_keep_count = -1
            pred_low_count = -1

        print(
            f"[Frame {self.idx}/{len(self.files)-1}] {fname} "
            f"GT={gt_count} "
            f"PRED_ALL={pred_all_count} "
            f"PRED_KEEP={pred_keep_count} "
            f"PRED_LOW={pred_low_count} "
            f"mode={self.args.mode} "
            f"filtering={self.args.filtering} "
            f"dist_filtering={self.args.dist_filtering} "
            f"score_filtering={self.args.score_filtering} "
            f"cluster={self.args.cluster}"
        )

    def next_frame(self, vis):
        self.idx = min(self.idx + 1, len(self.files) - 1)
        self.render_frame()
        return False

    def prev_frame(self, vis):
        self.idx = max(self.idx - 1, 0)
        self.render_frame()
        return False

    def run(self):
        self.vis.create_window(
            "GT / Prediction Viewer",
            self.args.window_width,
            self.args.window_height,
        )

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])
        opt.point_size = self.args.point_size

        self.vis.register_key_callback(ord("D"), self.next_frame)
        self.vis.register_key_callback(ord("A"), self.prev_frame)

        self.render_frame()

        self.vis.run()
        self.vis.destroy_window()


def main():
    args = parse_args()
    app = App(args)
    app.run()


if __name__ == "__main__":
    main()