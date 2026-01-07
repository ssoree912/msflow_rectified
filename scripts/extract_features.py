#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import default as default_cfg
from datasets import MVTecDataset, VisADataset, MSTCDataset
from datasets import MVTEC_CLASS_NAMES, VISA_CLASS_NAMES, MSTC_CLASS_NAMES
from models.extractors import build_extractor
from pruning.utils import apply_pruning_mask, resolve_pruning_mode, extractor_forward


class IndexedDataset(Dataset):
    def __init__(self, base, indices=None):
        self.base = base
        self.indices = indices
        self.x = getattr(base, "x", None)

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.base)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        x, y, _ = self.base[real_idx]
        return x, y, real_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute MSFlow features")
    parser.add_argument("--dataset", default="mstc", choices=["mvtec", "visa", "mstc"])
    parser.add_argument("--class-names", nargs="+", default=["all"])
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--data-path", default=None, help="override dataset root")
    parser.add_argument("--save-dir", default="./feature_cache")
    parser.add_argument("--layout", default=None, choices=["flat", "by-video"],
                        help="flat: save under save-dir/dataset/class/split. by-video: save under <save-dir>/<split>/<subdir>/<video_id>/")
    parser.add_argument("--save-subdir", default="features",
                        help="subdir name used with --layout by-video")
    parser.add_argument("--video-format", default="chunk", choices=["chunk", "npz", "npy"],
                        help="only for layout=by-video: chunk saves chunk_*.pt, npz saves one features.npz per video, npy saves one <vid>_res.npy per video")
    parser.add_argument("--npz-compress", action="store_true", default=False,
                        help="use np.savez_compressed for npz output")
    parser.add_argument("--video-suffix", default="_res.npy",
                        help="suffix for --video-format npy (default: _res.npy)")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="skip videos with existing cached files (by-video + npy/npz)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extractor", default=default_cfg.extractor)
    parser.add_argument("--pool-type", default=default_cfg.pool_type, choices=["avg", "max", "none"])
    parser.add_argument("--input-size", type=int, nargs=2, default=None)
    parser.add_argument("--save-dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="log progress every N batches")
    parser.add_argument("--pixel-eval", action="store_true", default=False,
                        help="load pixel masks if dataset supports it (slower)")
    parser.add_argument("--pruning-mode", default=default_cfg.pruning_mode,
                        choices=[
                            "dense",
                            "sparse",
                            "static",
                            "dynamic",
                            "reactivate_only",
                            "kill_only",
                            "kill_and_reactivate",
                        ])
    parser.add_argument("--pruning-sparsity", type=float, default=default_cfg.pruning_sparsity)
    parser.add_argument("--dwa-alpha", type=float, default=default_cfg.dwa_alpha)
    parser.add_argument("--dwa-beta", type=float, default=default_cfg.dwa_beta)
    parser.add_argument("--dwa-update-threshold", action="store_true", default=default_cfg.dwa_update_threshold)
    parser.add_argument("--dwa-threshold-percentile", type=int, default=default_cfg.dwa_threshold_percentile)
    return parser.parse_args()


def get_class_names(dataset, class_names):
    if class_names != ["all"]:
        return class_names
    if dataset == "mvtec":
        return MVTEC_CLASS_NAMES
    if dataset == "visa":
        return VISA_CLASS_NAMES
    return MSTC_CLASS_NAMES


def default_data_path(dataset):
    if dataset == "mvtec":
        return "./data/MVTec"
    if dataset == "visa":
        return "./data/VisA_pytorch/1cls"
    return "./data/shanghaitech"


def build_cfg(args, class_name):
    cfg = SimpleNamespace()
    cfg.dataset = args.dataset
    cfg.class_name = class_name
    cfg.data_path = args.data_path or default_data_path(args.dataset)
    cfg.img_mean = default_cfg.img_mean
    cfg.img_std = default_cfg.img_std
    cfg.extractor = args.extractor
    cfg.pool_type = args.pool_type
    cfg.pixel_eval = args.pixel_eval

    if args.input_size is not None:
        cfg.input_size = tuple(args.input_size)
    elif args.dataset == "mvtec":
        cfg.input_size = (256, 256) if class_name == "transistor" else (512, 512)
    elif args.dataset == "mstc":
        cfg.input_size = (256, 384)
    else:
        cfg.input_size = default_cfg.input_size

    cfg.pruning_mode = args.pruning_mode
    cfg.pruning_sparsity = args.pruning_sparsity
    cfg.dwa_alpha = args.dwa_alpha
    cfg.dwa_beta = args.dwa_beta
    cfg.dwa_update_threshold = args.dwa_update_threshold
    cfg.dwa_threshold_percentile = args.dwa_threshold_percentile
    cfg.pruning_type_value, cfg.pruning_forward_type = resolve_pruning_mode(cfg.pruning_mode)

    cfg.device = torch.device(args.device)
    return cfg


def build_pool(pool_type):
    if pool_type == "avg":
        return nn.AvgPool2d(3, 2, 1)
    if pool_type == "max":
        return nn.MaxPool2d(3, 2, 1)
    return nn.Identity()


def get_dataset(cfg, is_train):
    if cfg.dataset == "mvtec":
        return MVTecDataset(cfg, is_train=is_train)
    if cfg.dataset == "visa":
        return VisADataset(cfg, is_train=is_train)
    return MSTCDataset(cfg, is_train=is_train)


def relpath_or_abs(path, root):
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def flush_chunk(out_dir, chunk_idx, feats_accum, labels_accum, paths_accum, frame_idx_accum=None):
    if not labels_accum:
        return 0
    feats = [torch.cat(stage, dim=0) for stage in feats_accum]
    labels = torch.cat(labels_accum, dim=0)
    payload = {
        "features": feats,
        "labels": labels,
        "paths": paths_accum,
    }
    if frame_idx_accum is not None:
        payload["frame_indices"] = frame_idx_accum
    out_path = out_dir / f"chunk_{chunk_idx:06d}.pt"
    torch.save(payload, out_path)
    count = labels.shape[0]
    feats_accum[:] = [[] for _ in feats_accum]
    labels_accum[:] = []
    paths_accum[:] = []
    if frame_idx_accum is not None:
        frame_idx_accum[:] = []
    return count


def parse_frame_index(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    return 0


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def flush_video_npz(out_dir, vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, compress):
    if not labels_accum:
        return 0
    feats = [torch.cat(stage, dim=0).numpy() for stage in feats_accum]
    labels = torch.cat(labels_accum, dim=0).numpy()
    paths = np.asarray(paths_accum)
    frame_indices = np.asarray(frame_idx_accum, dtype=np.int32)
    payload = {
        "s1": feats[0],
        "s2": feats[1],
        "s3": feats[2],
        "labels": labels,
        "paths": paths,
        "frame_indices": frame_indices,
    }
    vdir = out_dir / vid
    vdir.mkdir(parents=True, exist_ok=True)
    out_path = vdir / "features.npz"
    if compress:
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)
    count = labels.shape[0]
    feats_accum[:] = [[] for _ in feats_accum]
    labels_accum[:] = []
    paths_accum[:] = []
    frame_idx_accum[:] = []
    return count


def flush_video_npy(out_dir, vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, suffix):
    if not labels_accum:
        return 0
    feats = [torch.cat(stage, dim=0).numpy() for stage in feats_accum]
    labels = torch.cat(labels_accum, dim=0).numpy()
    payload = {
        "s1": feats[0],
        "s2": feats[1],
        "s3": feats[2],
        "labels": labels,
        "paths": np.asarray(paths_accum),
        "frame_indices": np.asarray(frame_idx_accum, dtype=np.int32),
    }
    out_path = out_dir / f"{vid}{suffix}"
    np.save(out_path, payload, allow_pickle=True)
    count = labels.shape[0]
    feats_accum[:] = [[] for _ in feats_accum]
    labels_accum[:] = []
    paths_accum[:] = []
    frame_idx_accum[:] = []
    return count


def extract_split(cfg, args, split):
    is_train = split == "train"
    base_dataset = get_dataset(cfg, is_train=is_train)

    extractor, _ = build_extractor(cfg)
    apply_pruning_mask(extractor, cfg.pruning_sparsity)
    extractor = extractor.to(cfg.device).eval()
    pool_layer = build_pool(cfg.pool_type).to(cfg.device)

    save_dtype = torch.float16 if args.save_dtype == "fp16" else torch.float32
    if cfg.dataset == "mstc":
        split_dir = "training" if split == "train" else "testing"
    else:
        split_dir = split
    if args.layout == "flat":
        out_dir = Path(args.save_dir) / cfg.dataset / cfg.class_name / split
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.save_dir) / split_dir / args.save_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    prev_counts = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                prev_manifest = json.load(f)
            prev_counts = prev_manifest.get("video_counts", {}) or {}
        except Exception:
            prev_counts = {}
    skip_videos = set()
    indices = None
    if args.skip_existing and args.layout == "by-video" and args.video_format in ("npy", "npz"):
        if not getattr(base_dataset, "x", None):
            raise ValueError("--skip-existing requires dataset with file paths")
        vids = sorted({os.path.basename(os.path.dirname(p)) for p in base_dataset.x})
        for vid in vids:
            if args.video_format == "npy":
                out_path = out_dir / f"{vid}{args.video_suffix}"
            else:
                out_path = out_dir / vid / "features.npz"
            if out_path.exists():
                skip_videos.add(vid)
        if skip_videos:
            indices = [
                i for i, p in enumerate(base_dataset.x)
                if os.path.basename(os.path.dirname(p)) not in skip_videos
            ]
            if not indices:
                print(f"[{cfg.dataset}/{cfg.class_name}/{split}] all videos already cached; skipping.")
                return

    dataset = IndexedDataset(base_dataset, indices=indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    feats_accum = [[], [], []]
    labels_accum = []
    paths_accum = []
    frame_idx_accum = []
    chunk_idx = 0
    vid_chunk_idx = {}
    vid_seen = set()
    current_vid = None
    video_counts = dict(prev_counts)
    total = 0
    stage_shapes = None
    stage_channels = None
    start = time.time()
    seen = 0
    total_samples = len(dataset)
    total_batches = len(loader)
    log_interval = max(1, args.log_interval)

    for idx, batch in enumerate(loader):
        images, labels, indices = batch
        images = images.to(cfg.device, non_blocking=True)
        labels = labels.to(torch.uint8)

        with torch.no_grad():
            if args.amp and cfg.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    feats = extractor_forward(cfg, extractor, images)
            else:
                feats = extractor_forward(cfg, extractor, images)

            feats = [pool_layer(f) for f in feats]
            if stage_shapes is None:
                stage_shapes = [list(f.shape[1:]) for f in feats]
                stage_channels = [int(f.shape[1]) for f in feats]

        feats_cpu = [f.detach().to("cpu", dtype=save_dtype) for f in feats]
        labels_cpu = labels.cpu()
        seen += labels_cpu.shape[0]
        idx_list = indices.tolist()
        paths = [dataset.x[i] for i in idx_list]

        if (idx == 0) or ((idx + 1) % log_interval == 0) or ((idx + 1) == total_batches):
            elapsed = time.time() - start
            rate = seen / elapsed if elapsed > 0 else 0.0
            remain = total_samples - seen
            eta = format_seconds(remain / rate) if rate > 0 else "?:??"
            pct = (100.0 * seen / total_samples) if total_samples else 0.0
            print(
                f"[{cfg.dataset}/{cfg.class_name}/{split}] "
                f"{seen}/{total_samples} ({pct:.1f}%) "
                f"{rate:.1f} samples/s ETA {eta}"
            )

        if args.layout == "flat":
            feats_accum[0].append(feats_cpu[0])
            feats_accum[1].append(feats_cpu[1])
            feats_accum[2].append(feats_cpu[2])
            labels_accum.append(labels_cpu)
            paths_accum.extend([relpath_or_abs(p, cfg.data_path) for p in paths])
            if len(paths_accum) >= args.chunk_size:
                total += flush_chunk(out_dir, chunk_idx, feats_accum, labels_accum, paths_accum)
                chunk_idx += 1
            continue

        vid_list = [os.path.basename(os.path.dirname(p)) for p in paths]
        all_same = all(v == vid_list[0] for v in vid_list)
        if all_same:
            vid = vid_list[0]
            vid_seen.add(vid)
            if current_vid is None:
                current_vid = vid
            if vid != current_vid:
                if args.video_format == "npz":
                    count = flush_video_npz(out_dir, current_vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, args.npz_compress)
                    total += count
                    video_counts[current_vid] = count
                elif args.video_format == "npy":
                    count = flush_video_npy(out_dir, current_vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, args.video_suffix)
                    total += count
                    video_counts[current_vid] = count
                else:
                    vdir = out_dir / current_vid
                    vdir.mkdir(parents=True, exist_ok=True)
                    v_idx = vid_chunk_idx.get(current_vid, 0)
                    total += flush_chunk(vdir, v_idx, feats_accum, labels_accum, paths_accum, frame_idx_accum)
                    vid_chunk_idx[current_vid] = v_idx + 1
                current_vid = vid

            feats_accum[0].append(feats_cpu[0])
            feats_accum[1].append(feats_cpu[1])
            feats_accum[2].append(feats_cpu[2])
            labels_accum.append(labels_cpu)
            paths_accum.extend([relpath_or_abs(p, cfg.data_path) for p in paths])
            frame_idx_accum.extend([parse_frame_index(p) for p in paths])
            if args.video_format == "chunk" and len(paths_accum) >= args.chunk_size:
                vdir = out_dir / current_vid
                vdir.mkdir(parents=True, exist_ok=True)
                v_idx = vid_chunk_idx.get(current_vid, 0)
                total += flush_chunk(vdir, v_idx, feats_accum, labels_accum, paths_accum, frame_idx_accum)
                vid_chunk_idx[current_vid] = v_idx + 1
            continue

        for i, vid in enumerate(vid_list):
            vid_seen.add(vid)
            if current_vid is None:
                current_vid = vid
            if vid != current_vid:
                if args.video_format == "npz":
                    count = flush_video_npz(out_dir, current_vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, args.npz_compress)
                    total += count
                    video_counts[current_vid] = count
                elif args.video_format == "npy":
                    count = flush_video_npy(out_dir, current_vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, args.video_suffix)
                    total += count
                    video_counts[current_vid] = count
                else:
                    vdir = out_dir / current_vid
                    vdir.mkdir(parents=True, exist_ok=True)
                    v_idx = vid_chunk_idx.get(current_vid, 0)
                    total += flush_chunk(vdir, v_idx, feats_accum, labels_accum, paths_accum, frame_idx_accum)
                    vid_chunk_idx[current_vid] = v_idx + 1
                current_vid = vid

            feats_accum[0].append(feats_cpu[0][i:i+1])
            feats_accum[1].append(feats_cpu[1][i:i+1])
            feats_accum[2].append(feats_cpu[2][i:i+1])
            labels_accum.append(labels_cpu[i:i+1])
            paths_accum.append(relpath_or_abs(paths[i], cfg.data_path))
            frame_idx_accum.append(parse_frame_index(paths[i]))

            if args.video_format == "chunk" and len(paths_accum) >= args.chunk_size:
                vdir = out_dir / current_vid
                vdir.mkdir(parents=True, exist_ok=True)
                v_idx = vid_chunk_idx.get(current_vid, 0)
                total += flush_chunk(vdir, v_idx, feats_accum, labels_accum, paths_accum, frame_idx_accum)
                vid_chunk_idx[current_vid] = v_idx + 1

    if args.layout == "flat":
        total += flush_chunk(out_dir, chunk_idx, feats_accum, labels_accum, paths_accum)
    else:
        if current_vid is not None and labels_accum:
            if args.video_format == "npz":
                count = flush_video_npz(out_dir, current_vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, args.npz_compress)
                total += count
                video_counts[current_vid] = count
            elif args.video_format == "npy":
                count = flush_video_npy(out_dir, current_vid, feats_accum, labels_accum, paths_accum, frame_idx_accum, args.video_suffix)
                total += count
                video_counts[current_vid] = count
            else:
                vdir = out_dir / current_vid
                vdir.mkdir(parents=True, exist_ok=True)
                v_idx = vid_chunk_idx.get(current_vid, 0)
                total += flush_chunk(vdir, v_idx, feats_accum, labels_accum, paths_accum, frame_idx_accum)
    elapsed = time.time() - start

    manifest = {
        "dataset": cfg.dataset,
        "class_name": cfg.class_name,
        "split": split,
        "num_samples": total,
        "pool_type": cfg.pool_type,
        "input_size": list(cfg.input_size),
        "extractor": cfg.extractor,
        "save_dtype": args.save_dtype,
        "stage_shapes": stage_shapes,
        "stage_channels": stage_channels,
        "chunk_size": args.chunk_size,
    }
    manifest["layout"] = args.layout
    manifest["video_format"] = args.video_format
    manifest["npz_compress"] = args.npz_compress
    manifest["video_suffix"] = args.video_suffix
    if args.layout == "by-video":
        manifest["save_subdir"] = args.save_subdir
        manifest["video_ids"] = sorted(vid_seen)
        if video_counts:
            manifest["video_counts"] = video_counts
        if skip_videos:
            manifest["skipped_videos"] = sorted(skip_videos)
            manifest["skipped_count"] = len(skip_videos)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    skip_note = f", skipped {len(skip_videos)} videos" if skip_videos else ""
    print(f"[{cfg.dataset}/{cfg.class_name}/{split}] saved {total} samples in {elapsed:.1f}s -> {out_dir}{skip_note}")


def main():
    args = parse_args()
    if args.layout is None:
        args.layout = "by-video" if args.dataset == "mstc" else "flat"
    if args.layout == "by-video" and args.dataset != "mstc":
        raise ValueError("--layout by-video is only supported for mstc")
    if args.layout == "flat" and args.video_format != "chunk":
        raise ValueError("--video-format npz/npy requires --layout by-video")
    class_names = get_class_names(args.dataset, args.class_names)
    splits = ["train", "test"] if args.split == "all" else [args.split]

    for cls in class_names:
        cfg = build_cfg(args, cls)
        for split in splits:
            extract_split(cfg, args, split)


if __name__ == "__main__":
    main()
