
import os, time, datetime, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import default as cfg
from datasets import MVTecDataset, VisADataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from utils import positionalencoding2d, load_weights, save_weights, t2np, infer_stage_hw
from pair_cutpaste_dataset import PairCutPasteMVTec, PairCutPasteVisA
from models.velocity import Velocity3Stage
from post_process import post_process
from evaluations import eval_det_loc
from utils import Score_Observer, positionalencoding2d, t2np

def model_forward_features(c, extractor, image):
    # Return the 3 pre-flow feature maps from extractor
    extractor = extractor.eval()
    with torch.no_grad():
        feats = extractor(image)
        # The provided resnet wrapper is expected to return [x1,x2,x3]
        if isinstance(feats, (list, tuple)):
            return feats[0], feats[1], feats[2]
        else:
            raise RuntimeError("Extractor did not return a 3-tuple of stage features")


def inference_with_velocity(c, test_loader, extractor, parallel_flows, fusion_flow,
                            vel_model, alpha_list):
    # put models in eval
    parallel_flows = [pf.eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.eval()
    if vel_model is not None:
        vel_model = vel_model.eval()

    gt_label_list = []
    gt_mask_list = []
    outputs_list = [list() for _ in parallel_flows]      # original msflow logp per stage
    outputs_list_diff = [list() for _ in parallel_flows] # velocity-corrected diffs per stage
    size_list = []

    start = time.time()
    with torch.no_grad():
        for idx, (image, label, mask) in enumerate(test_loader):
            image = image.to(c.device)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            # ------------------------------------------------
            # 1) feature extraction (shared by raw/velocity)
            # ------------------------------------------------
            h_list = extractor(image)   # (x1,x2,x3) before pooling
            if c.pool_type == 'avg':
                pool_layer = nn.AvgPool2d(3, 2, 1)
            elif c.pool_type == 'max':
                pool_layer = nn.MaxPool2d(3, 2, 1)
            else:
                pool_layer = nn.Identity()

            # ------------------------------------------------
            # 2) velocity correction on ALL stages (concat raw+corr through flows)
            # ------------------------------------------------
            if vel_model is not None and alpha_list is not None:
                # start from original features
                x1, x2, x3 = h_list
                B = image.shape[0]
                K = getattr(c, 'vel_steps', 1)         # e.g., 5~20
                dt = 1.0 / K
                for k in range(K):
                    # time at the *current* state
                    t = torch.full((x1.size(0), 1), k * dt, device=x1.device)
                    d1, d2, d3 = vel_model((x1, x2, x3), t)
                    x1 = x1 + dt * d1 * alpha_list[0]
                    x2 = x2 + dt * d2 * alpha_list[1]
                    x3 = x3 + dt * d3 * alpha_list[2]
                corrected_feats = (x1, x2, x3)

                z_cat_list = []
                for (h_raw, h_corr, flow, c_cond) in zip(h_list, corrected_feats, parallel_flows, c.c_conds):
                    y_raw = pool_layer(h_raw)
                    y_corr = pool_layer(h_corr)
                    _, _, H, W = y_raw.shape
                    cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                    y_cat = torch.cat([y_raw, y_corr], dim=0)
                    cond_cat = torch.cat([cond, cond], dim=0)
                    z_cat, _ = flow(y_cat, [cond_cat])
                    z_cat_list.append(z_cat)

                z_fused_cat_list, _ = fusion_flow(z_cat_list)

                for lvl, z_fused_cat in enumerate(z_fused_cat_list):
                    z_raw, z_corr = torch.split(z_fused_cat, B, dim=0)
                    if idx == 0:
                        size_list.append(list(z_raw.shape[-2:]))
                    logp_raw = -0.5 * torch.mean(z_raw ** 2, 1)
                    logp_corr = -0.5 * torch.mean(z_corr ** 2, 1)
                    outputs_list[lvl].append(logp_raw)
                    diff = logp_raw - logp_corr
                    # post_process는 "클수록 정상(logp)" 
                    # diff>0(교정 후 더 이상해짐)인 부분만 anomaly 신호로 쓰기 위해 음수로 페널티를 준다.
                    pseudo_logp = -torch.relu(diff)  # diff<=0 -> 0, diff>0 -> negative
                    outputs_list_diff[lvl].append(pseudo_logp)
            else:
                # normal MSFlow forward: extractor -> per-stage flow -> fusion
                z_list = []
                for (h, flow, c_cond) in zip(h_list, parallel_flows, c.c_conds):
                    y = pool_layer(h)
                    B, _, H, W = y.shape
                    cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                    z, _ = flow(y, [cond])
                    z_list.append(z)

                z_list, _ = fusion_flow(z_list)

                for lvl, z in enumerate(z_list):
                    if idx == 0:
                        size_list.append(list(z.shape[-2:]))
                    logp = -0.5 * torch.mean(z ** 2, 1)
                    outputs_list[lvl].append(logp)

                for lvl in range(len(parallel_flows)):
                    zeros = torch.zeros_like(outputs_list[lvl][-1])
                    outputs_list_diff[lvl].append(zeros)

    fps = len(test_loader.dataset) / (time.time() - start)
    print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
          f"inference_with_velocity done. FPS: {fps:.1f}")

    return gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff

# def train_velocity_one_epoch(c, loader, extractor, vel_model, opt_list, alpha_list, scaler=None, amp=False, device="cuda"):
#     vel_model.train()
#     loss_meter = 0.0
#     count = 0
#     l1 = nn.SmoothL1Loss()

#     for (x_clean, x_abn, _, _) in loader:
#         x_clean = x_clean.to(device, non_blocking=True)
#         x_abn   = x_abn.to(device, non_blocking=True)

#         # Get stage features
#         with torch.no_grad():
#             xc1, xc2, xc3 = extractor(x_clean)
#             xa1, xa2, xa3 = extractor(x_abn)

#         # Forward velocity nets
#         opt1, opt2, opt3 = opt_list
#         for opt in opt_list: opt.zero_grad(set_to_none=True)

#         if amp and scaler:
#             with torch.cuda.amp.autocast():
#                 d1, d2, d3 = vel_model((xa1, xa2, xa3))
#                 xa1_p = xa1 + alpha_list[0] * d1
#                 xa2_p = xa2 + alpha_list[1] * d2
#                 xa3_p = xa3 + alpha_list[2] * d3
#                 loss = l1(xc1, xa1_p) + l1(xc2, xa2_p) + l1(xc3, xa3_p)
#             scaler.scale(loss).backward()
#             for opt in opt_list: scaler.step(opt)
#             scaler.update()
#         else:
#             d1, d2, d3 = vel_model((xa1, xa2, xa3))
#             xa1_p = xa1 + alpha_list[0] * d1
#             xa2_p = xa2 + alpha_list[1] * d2
#             xa3_p = xa3 + alpha_list[2] * d3
#             loss = l1(xc1, xa1_p) + l1(xc2, xa2_p) + l1(xc3, xa3_p)
#             loss.backward()
#             for opt in opt_list: opt.step()

#         bs = x_clean.shape[0]
#         loss_meter += loss.item() * bs
#         count += bs

#     return loss_meter / max(1, count)

# --- find train_velocity_one_epoch and replace its body with: ---

def train_velocity_one_epoch(c, loader, extractor, vel_model, opt_list, alpha_list, scaler=None, amp=False, device="cuda"):
    vel_model.train()
    l2 = torch.nn.MSELoss()
    loss_meter, count = 0.0, 0

    for (x_clean, x_abn, *_) in loader:
        x_clean = x_clean.to(device, non_blocking=True)
        x_abn   = x_abn.to(device, non_blocking=True)
        B = x_clean.size(0)

        # frozen feature extractor
        with torch.no_grad():
            xc1, xc2, xc3 = extractor(x_clean)  # "normal" features = z1
            xa1, xa2, xa3 = extractor(x_abn)    # "abnormal" features = z0

        # sample t ~ U[0,1], broadcastable to feature maps
        t = torch.rand(B, 1, device=device)
        t4 = t.view(B, 1, 1, 1)

        # build interpolants z_t = z0 + t*(z1 - z0) at each stage
        zt1 = xa1 + t4 * (xc1 - xa1)
        zt2 = xa2 + t4 * (xc2 - xa2)
        zt3 = xa3 + t4 * (xc3 - xa3)

        # targets: straight-line velocities (z1 - z0)
        v1 = (xc1 - xa1)
        v2 = (xc2 - xa2)
        v3 = (xc3 - xa3)

        for opt in opt_list: opt.zero_grad(set_to_none=True)

        if amp and scaler:
            with torch.cuda.amp.autocast():
                d1, d2, d3 = vel_model((zt1, zt2, zt3), t)  # time-conditioned
                loss = l2(d1, v1) + l2(d2, v2) + l2(d3, v3)
            scaler.scale(loss).backward()
            for opt in opt_list: scaler.step(opt)
            scaler.update()
        else:
            d1, d2, d3 = vel_model((zt1, zt2, zt3), t)
            loss = l2(d1, v1) + l2(d2, v2) + l2(d3, v3)
            loss.backward()
            for opt in opt_list: opt.step()

        bs = x_clean.size(0)
        loss_meter += loss.item() * bs
        count += bs

    return loss_meter / max(1, count)

def anomaly_score_from_mulmap(c, anomaly_score_map_mul_np: np.ndarray):
    """
    anomaly_score_map_mul_np: (B,H,W) numpy, 값이 클수록 이상.
    post_process 내부처럼 top-k 평균으로 per-image anomaly score 산출.
    """
    B, H, W = anomaly_score_map_mul_np.shape
    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)
    top_k = max(1, min(top_k, H * W))
    flat = anomaly_score_map_mul_np.reshape(B, -1)

    # top-k largest mean (fast)
    topk = np.partition(flat, -top_k, axis=1)[:, -top_k:]
    return topk.mean(axis=1)

def default_velocity_cfg(c_list):
    # Provide small defaults per stage (can be overridden by CLI)
    return {
        "s1_hidden": 128, "s1_resblocks": 2,
        "s2_hidden": 128, "s2_resblocks": 2,
        "s3_hidden": 128, "s3_resblocks": 2,
    }

def find_msflow_ckpt(cfg, dataset_name, class_name):
    # Example: work_dirs/msflow_wide_resnet50_2_avgpool_pl258/mvtec/bottle/best_loc_auroc.pt
    model_dir = "msflow_wide_resnet50_2_avgpool_pl258"
    ckpt_path = os.path.join(cfg.work_dir, model_dir, dataset_name, class_name, "best_loc_auroc.pt")
    return ckpt_path

def main():
    parser = argparse.ArgumentParser("Train velocity nets (pre-flow) with CutPaste-Scar pairs")
    parser.add_argument('--dataset', default='mvtec', type=str, choices=['mvtec', 'visa'], help='dataset')
    parser.add_argument('--class-names', default=['all'], type=str, nargs='+', help='class names')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    parser.add_argument('--vel-epochs', default=40, type=int, help='epochs for velocity nets')
    parser.add_argument('--vel-lr1', default=1e-4, type=float, help='lr for stage1 velocity')
    parser.add_argument('--vel-lr2', default=1e-4, type=float, help='lr for stage2 velocity')
    parser.add_argument('--vel-lr3', default=1e-4, type=float, help='lr for stage3 velocity')
    parser.add_argument('--alpha1', default=1.0, type=float, help='step factor for stage1')
    parser.add_argument('--alpha2', default=1.0, type=float, help='step factor for stage2')
    parser.add_argument('--alpha3', default=1.0, type=float, help='step factor for stage3')
    parser.add_argument('--save-root', default='./work_dirs/velocity_preflow', type=str, help='save dir root')
    parser.add_argument('--amp-enable', action='store_true', default=False, help='use torch.amp')
    args = parser.parse_args()

    # Honor defaults from default.py
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build extractor + get stage channels
    extractor, c_list = build_extractor(cfg)  # pass default.py namespace

    # Build dummy flows and load saved MSFlow weights (aligns code & ensures no mismatch)
    stage_hw = infer_stage_hw(cfg, extractor)
    parallel_flows, fusion_flow = build_msflow_model(cfg, c_list, stage_hw)
    # We'll load per-class inside the loop below.

    for k, v in vars(args).items():
        setattr(cfg, k, v)
    # Training for each class
        
    if cfg.dataset == 'mvtec':
        from datasets import MVTEC_CLASS_NAMES
        setattr(cfg, 'data_path', './data/MVTec')
        if cfg.class_names == ['all']:
            setattr(cfg, 'class_names', MVTEC_CLASS_NAMES)
    elif cfg.dataset == 'visa':
        from datasets import VISA_CLASS_NAMES
        setattr(cfg, 'data_path', './data/VisA_pytorch/1cls')
        if cfg.class_names == ['all']:
            setattr(cfg, 'class_names', VISA_CLASS_NAMES)

        
    cfg.input_size = (256, 256) if cfg.class_name == 'transistor' else (512, 512)

    extractor = extractor.to(device).eval()  # frozen backbone
    parallel_flows = [pf.to(device).eval() for pf in parallel_flows]
    fusion_flow = fusion_flow.to(device).eval()

    for cls in cfg.class_names:
        cfg.class_name = cls

        if cfg.dataset == 'mvtec':
            test_set = MVTecDataset(cfg, is_train=False)
        elif cfg.dataset == 'visa':
            test_set = VisADataset(cfg, is_train=False)
        else:
            raise ValueError("velocity training currently supports mvtec/visa")
            
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Try to load per-class MSFlow checkpoint (if available)
        ckpt_path = find_msflow_ckpt(cfg, args.dataset, cls)
        if os.path.exists(ckpt_path):
            try:
                load_weights(parallel_flows, fusion_flow, ckpt_path)
            except Exception as e:
                print(f"Warning: could not load MSFlow weights for {cls}: {e}")
        else:
            print(f"MSFlow checkpoint not found for class {cls}: {ckpt_path}")


        # If you want to actually load, uncomment the following lines and set a proper path:
        # ckpt_path = find_msflow_ckpt(work_dir, extractor_name=extractor.__class__.__name__.lower(),
        #                              dataset_name=args.dataset, class_name=cls)
        # if os.path.exists(ckpt_path):
        #     load_weights(parallel_flows, fusion_flow, ckpt_path)

        # Dataset + loader
        if cfg.dataset == 'mvtec':
            train_set = PairCutPasteMVTec(cfg, cls)
        elif cfg.dataset == 'visa':
            train_set = PairCutPasteVisA(cfg, cls)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Build velocity model (3 stages) with independent hyperparams
        vel_cfg = default_velocity_cfg(c_list)
        vel_model = Velocity3Stage(c_list, vel_cfg).to(device)

        # Separate optimizers per stage (so you can set different hparams later)
        p1 = list(vel_model.stage1.parameters())
        p2 = list(vel_model.stage2.parameters())
        p3 = list(vel_model.stage3.parameters())
        opt1 = torch.optim.AdamW(p1, lr=args.vel_lr1, weight_decay=1e-4)
        opt2 = torch.optim.AdamW(p2, lr=args.vel_lr2, weight_decay=1e-4)
        opt3 = torch.optim.AdamW(p3, lr=args.vel_lr3, weight_decay=1e-4)

        scaler = torch.cuda.amp.GradScaler() if (args.amp_enable and device.type=='cuda') else None

        save_dir = os.path.join(args.save_root, args.dataset, cls)
        os.makedirs(save_dir, exist_ok=True)

        best_loc_auroc = 0.0
        last_fps = None
        best_det_fps = None
        best_loc_fps = None
        det_obs = None
        loc_obs = None
        pro_obs = None
        
        for ep in range(args.vel_epochs):
            loss = train_velocity_one_epoch(
                globals(), loader, extractor, vel_model,
                opt_list=[opt1,opt2,opt3],
                alpha_list=[args.alpha1,args.alpha2,args.alpha3],
                scaler=scaler, amp=args.amp_enable, device=str(device)
            )
            print(f"... Epoch {ep} vel_loss={loss:.4f}")

            # ---- EVALUATE (velocity 반영) ----
            eval_start = time.time()
            gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff = \
                inference_with_velocity(
                    cfg, test_loader, extractor, parallel_flows, fusion_flow,
                    vel_model, [args.alpha1, args.alpha2, args.alpha3]
                )
            last_fps = len(test_loader.dataset) / (time.time() - eval_start)

            # --- RAW (msflow) ---
            anomaly_score_raw, anomaly_score_map_add_raw, anomaly_score_map_mul_raw = \
                post_process(cfg, size_list, outputs_list)

            # --- DIFF (velocity pseudo-logp) ---
            _, anomaly_score_map_add_diff, anomaly_score_map_mul_diff = \
                post_process(cfg, size_list, outputs_list_diff)

            # --- FINAL (raw + diff) ---
            anomaly_score_map_add_final = anomaly_score_map_add_raw + anomaly_score_map_add_diff
            anomaly_score_map_mul_final = anomaly_score_map_mul_raw + anomaly_score_map_mul_diff

            # DET도 velocity 반영: FINAL mul-map으로 anomaly_score 재계산
            anomaly_score_final = anomaly_score_from_mulmap(cfg, anomaly_score_map_mul_final)

            # make score observers like train.py
            if ep == 0:
                det_obs = Score_Observer('Det.AUROC', args.vel_epochs)
                loc_obs = Score_Observer('Loc.AUROC', args.vel_epochs)
                pro_obs = Score_Observer('Loc.PRO', args.vel_epochs)

            det_auroc, loc_auroc, loc_pro_auc, \
                best_det_flag, best_loc_flag, best_pro_flag = \
                    eval_det_loc(
                        det_obs, loc_obs, pro_obs, ep,
                        gt_label_list, anomaly_score_final,
                        gt_mask_list, anomaly_score_map_add_final,
                        anomaly_score_map_mul_final,
                        pro_eval=False
                    )
            if best_det_flag:
                best_det_fps = last_fps
            if best_loc_flag:
                best_loc_fps = last_fps

            # ---- SAVE last (as before) ----
            state = {
                'epoch': ep,
                'model': vel_model.state_dict(),
                'alpha': [args.alpha1,args.alpha2,args.alpha3],
                'cfg': vel_cfg,
            }
            torch.save(state, os.path.join(save_dir, "velocity_last.pt"))

            # ---- SAVE best by Loc.AUROC ----
            if loc_auroc > best_loc_auroc:
                best_loc_auroc = loc_auroc
                torch.save(state, os.path.join(save_dir, "velocity_best_loc.pt"))
                print(f"[{cls}] New best Loc.AUROC: {loc_auroc:.2f}, saved velocity_best_loc.pt")

        if det_obs is not None and loc_obs is not None and last_fps is not None:
            if best_det_fps is None:
                best_det_fps = last_fps
            if best_loc_fps is None:
                best_loc_fps = last_fps
            out_path = os.path.join(save_dir, 'best_metrics.csv')
            with open(out_path, 'w') as f:
                f.write('dataset,class_name,best_det_auroc,best_det_epoch,best_det_fps,best_loc_auroc,best_loc_epoch,best_loc_fps\n')
                f.write(
                    f'{args.dataset},{cls},'
                    f'{det_obs.max_score:.4f},{det_obs.max_epoch},'
                    f'{best_det_fps:.4f},'
                    f'{loc_obs.max_score:.4f},{loc_obs.max_epoch},'
                    f'{best_loc_fps:.4f}\n'
                )
            print(f"Saved best metrics to {out_path}")

if __name__ == "__main__":
    main()
