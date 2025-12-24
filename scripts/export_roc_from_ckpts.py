#!/usr/bin/env python3
import argparse
import copy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import default as default_cfg
from datasets import MVTecDataset, VisADataset, MSTCDataset
from datasets import MVTEC_CLASS_NAMES, VISA_CLASS_NAMES, MSTC_CLASS_NAMES
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from models.velocity import Velocity3Stage
from post_process import post_process
from train import model_forward
from train_velocity import inference_with_velocity, default_velocity_cfg
from utils import load_weights, t2np


def make_cfg():
    keys = [
        "extractor",
        "pool_type",
        "c_conds",
        "parallel_blocks",
        "clamp_alpha",
        "img_mean",
        "img_std",
        "top_k",
        "batch_size",
        "workers",
        "work_dir",
    ]
    cfg = SimpleNamespace()
    for k in keys:
        setattr(cfg, k, getattr(default_cfg, k))
    return cfg


def set_dataset_cfg(cfg, dataset, class_name, data_path, device):
    cfg.dataset = dataset
    cfg.class_name = class_name
    cfg.data_path = data_path
    if dataset == "mvtec":
        cfg.input_size = (256, 256) if class_name == "transistor" else (512, 512)
    elif dataset == "mstc":
        cfg.input_size = (256, 384)
    else:
        cfg.input_size = (512, 512)
    cfg.device = device


def get_class_names(dataset, class_names):
    if class_names != ["all"]:
        return class_names
    if dataset == "mvtec":
        return MVTEC_CLASS_NAMES
    if dataset == "visa":
        return VISA_CLASS_NAMES
    if dataset == "mstc":
        return MSTC_CLASS_NAMES
    raise ValueError(f"Unknown dataset: {dataset}")


def get_dataset(cfg, is_train=False):
    if cfg.dataset == "mvtec":
        return MVTecDataset(cfg, is_train=is_train)
    if cfg.dataset == "visa":
        return VisADataset(cfg, is_train=is_train)
    if cfg.dataset == "mstc":
        return MSTCDataset(cfg, is_train=is_train)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def compute_metrics(gt_label_list, gt_mask_list, anomaly_score, anomaly_score_map_add):
    gt_label = np.asarray(gt_label_list, dtype=bool)
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
    det_auroc = roc_auc_score(gt_label, anomaly_score) * 100
    loc_auroc = roc_auc_score(gt_mask.flatten(), anomaly_score_map_add.flatten()) * 100
    return det_auroc, loc_auroc


def eval_msflow(cfg, ckpt_path, batch_size, num_workers):
    extractor, output_channels = build_extractor(cfg)
    parallel_flows, fusion_flow = build_msflow_model(cfg, output_channels)

    load_weights(parallel_flows, fusion_flow, ckpt_path)

    device = cfg.device
    extractor = extractor.to(device).eval()
    parallel_flows = [pf.to(device).eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.to(device).eval()

    test_set = get_dataset(cfg, is_train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    gt_label_list, gt_mask_list = [], []
    start = time.time()
    with torch.no_grad():
        for idx, (image, label, mask) in enumerate(test_loader):
            image = image.to(device)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            z_list, _ = model_forward(cfg, extractor, parallel_flows, fusion_flow, image)
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                logp = -0.5 * torch.mean(z ** 2, 1)
                outputs_list[lvl].append(logp)

    fps = len(test_set) / (time.time() - start)
    anomaly_score, anomaly_score_map_add, _ = post_process(cfg, size_list, outputs_list)
    det_auroc, loc_auroc = compute_metrics(gt_label_list, gt_mask_list, anomaly_score, anomaly_score_map_add)
    return det_auroc, loc_auroc, fps


def eval_velocity(cfg, msflow_ckpt, vel_ckpt, batch_size, num_workers):
    extractor, c_list = build_extractor(cfg)
    parallel_flows, fusion_flow = build_msflow_model(cfg, c_list)

    load_weights(parallel_flows, fusion_flow, msflow_ckpt)

    device = cfg.device
    extractor = extractor.to(device).eval()
    parallel_flows = [pf.to(device).eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.to(device).eval()

    state = torch.load(vel_ckpt, map_location=device)
    vel_cfg = state.get("cfg", default_velocity_cfg(c_list))
    vel_model = Velocity3Stage(c_list, vel_cfg).to(device)
    vel_model.load_state_dict(state["model"])
    alpha_list = state.get("alpha", [1.0, 1.0, 1.0])

    test_set = get_dataset(cfg, is_train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    start = time.time()
    gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff = inference_with_velocity(
        cfg, test_loader, extractor, parallel_flows, fusion_flow, vel_model, alpha_list
    )
    fps = len(test_set) / (time.time() - start)

    anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(cfg, size_list, outputs_list)
    _, anomaly_score_map_add_diff, anomaly_score_map_mul_diff = post_process(cfg, size_list, outputs_list_diff)
    anomaly_score_map_add = anomaly_score_map_add + anomaly_score_map_add_diff
    anomaly_score_map_mul = anomaly_score_map_mul + anomaly_score_map_mul_diff
    h, w = cfg.input_size
    top_k = max(1, int(h * w * cfg.top_k))
    mul_map_t = torch.from_numpy(anomaly_score_map_mul).to(torch.float32)
    bsz = mul_map_t.shape[0]
    anomaly_score = mul_map_t.view(bsz, -1).topk(top_k, dim=-1)[0].mean(dim=-1).cpu().numpy()
    det_auroc, loc_auroc = compute_metrics(gt_label_list, gt_mask_list, anomaly_score, anomaly_score_map_add)
    return det_auroc, loc_auroc, fps


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "method",
        "dataset",
        "class_name",
        "ckpt_name",
        "ckpt_path",
        "det_auroc",
        "loc_auroc",
        "fps",
    ]
    with out_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in header) + "\n")


def write_json(rows, out_path):
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Export ROC metrics from MSFlow/velocity checkpoints")
    parser.add_argument("--dataset", default="mvtec", choices=["mvtec", "visa", "mstc"])
    parser.add_argument("--class-names", nargs="+", default=["all"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--msflow-root", default=None)
    parser.add_argument("--msflow-ckpt-name", default="best_loc_auroc.pt")
    parser.add_argument("--velocity-roots", nargs="*", default=[])
    parser.add_argument("--velocity-ckpt-name", default="velocity_best_loc.pt")
    parser.add_argument("--out-csv", default="./work_dirs/roc_metrics.csv")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    if args.data_path is None:
        if args.dataset == "mvtec":
            data_path = "./data/MVTec"
        elif args.dataset == "visa":
            data_path = "./data/VisA_pytorch/1cls"
        else:
            data_path = "./data/shanghaitech"
    else:
        data_path = args.data_path

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    base_cfg = make_cfg()
    rows = []
    class_names = get_class_names(args.dataset, args.class_names)

    if args.msflow_root:
        msflow_root = Path(args.msflow_root)
        method_label = msflow_root.name
        for cls in class_names:
            cfg = copy.deepcopy(base_cfg)
            set_dataset_cfg(cfg, args.dataset, cls, data_path, device)
            ckpt_path = msflow_root / args.dataset / cls / args.msflow_ckpt_name
            if not ckpt_path.exists():
                continue
            det_auroc, loc_auroc, fps = eval_msflow(cfg, str(ckpt_path), args.batch_size, args.num_workers)
            rows.append(
                {
                    "method": method_label,
                    "dataset": args.dataset,
                    "class_name": cls,
                    "ckpt_name": args.msflow_ckpt_name,
                    "ckpt_path": str(ckpt_path),
                    "det_auroc": round(det_auroc, 4),
                    "loc_auroc": round(loc_auroc, 4),
                    "fps": round(fps, 4),
                }
            )

    for vel_root in args.velocity_roots:
        vel_root = Path(vel_root)
        method_label = vel_root.name
        for cls in class_names:
            cfg = copy.deepcopy(base_cfg)
            set_dataset_cfg(cfg, args.dataset, cls, data_path, device)
            vel_ckpt = vel_root / args.dataset / cls / args.velocity_ckpt_name
            if not vel_ckpt.exists():
                continue
            if not args.msflow_root:
                raise ValueError("velocity evaluation requires --msflow-root for base flow weights")
            msflow_ckpt = Path(args.msflow_root) / args.dataset / cls / args.msflow_ckpt_name
            if not msflow_ckpt.exists():
                continue
            det_auroc, loc_auroc, fps = eval_velocity(
                cfg,
                str(msflow_ckpt),
                str(vel_ckpt),
                args.batch_size,
                args.num_workers,
            )
            rows.append(
                {
                    "method": method_label,
                    "dataset": args.dataset,
                    "class_name": cls,
                    "ckpt_name": args.velocity_ckpt_name,
                    "ckpt_path": str(vel_ckpt),
                    "det_auroc": round(det_auroc, 4),
                    "loc_auroc": round(loc_auroc, 4),
                    "fps": round(fps, 4),
                }
            )

    out_csv = Path(args.out_csv)
    write_csv(rows, out_csv)
    if args.out_json:
        write_json(rows, Path(args.out_json))
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
