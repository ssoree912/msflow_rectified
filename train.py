import os
import time
import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from datasets import MVTecDataset, VisADataset, MSTCDataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from models.velocity import Velocity3Stage
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc


def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    h_list = extractor(image)
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(parallel_jac_list)

    return z_list, jac

def train_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler=None):
    parallel_flows = [parallel_flow.train() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.train()

    for sub_epoch in range(c.sub_epochs):
        epoch_loss = 0.
        image_count = 0
        for idx, (image, _, _) in enumerate(loader):
            optimizer.zero_grad()
            image = image.to(c.device)
            if scaler:
                with autocast():
                    z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 2)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                loss = 0.
                for z in z_list:
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                loss = loss - jac
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2)
                optimizer.step()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if warmup_scheduler:
            warmup_scheduler.step()
        if decay_scheduler:
            decay_scheduler.step()

        mean_epoch_loss = epoch_loss / image_count
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                epoch, sub_epoch, mean_epoch_loss, lr))
        

def inference_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, vel_model=None, alpha_list=None):
    parallel_flows = [parallel_flow.eval() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.eval()
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    gt_mask_list = list()
    outputs_list = [list() for _ in parallel_flows]
    outputs_list_diff = [list() for _ in parallel_flows]
    size_list = []
    start = time.time()
    with torch.no_grad():
        for idx, (image, label, mask) in enumerate(loader):
            image = image.to(c.device)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                # Original logp map
                logp = - 0.5 * torch.mean(z**2, 1)
                outputs_list[lvl].append(logp)

                # If velocity model is available, compute corrected logp
                # if vel_model is not None and alpha_list is not None:
                #     # Recompute on corrected features for this stage
                #     # Extract fresh pre-flow features for this image
                #     h_list = extractor(image)
                #     x1, x2, x3 = h_list
                #     K = 5
                #     a1 = alpha_list[0] / K
                #     a2 = alpha_list[1] / K
                #     a3 = alpha_list[2] / K
                #     for _ in range(K):
                #         d1, d2, d3 = vel_model((x1, x2, x3))
                #         x1 = x1 + a1 * d1
                #         x2 = x2 + a2 * d2
                #         x3 = x3 + a3 * d3

                #     xa_p = (x1, x2, x3)[lvl]

                #     if c.pool_type == 'avg':
                #         pool_layer = nn.AvgPool2d(3, 2, 1)
                #     elif c.pool_type == 'max':
                #         pool_layer = nn.MaxPool2d(3, 2, 1)
                #     else:
                #         pool_layer = nn.Identity()

                #     y_corr = pool_layer(xa_p)
                #     Bc, _, Hc, Wc = y_corr.shape
                #     cond_corr = positionalencoding2d(c.c_conds[lvl], Hc, Wc).to(c.device).unsqueeze(0).repeat(Bc, 1, 1, 1)
                #     z_corr, _ = parallel_flows[lvl](y_corr, [cond_corr, ])
                #     logp_corr = - 0.5 * torch.mean(z_corr**2, 1)
                #     # Store difference map: logp(test) - logp(normalized)
                #     outputs_list_diff[lvl].append(logp - logp_corr)

                # Accumulate NLL loss as before
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))

            loss = loss - jac
            loss = loss.mean()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]

        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                epoch, mean_epoch_loss, fps))

    return gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff



def train(c):
    
    if c.wandb_enable:
        wandb.finish()
        wandb_run = wandb.init(
            project='65001-msflow', 
            group=c.version_name,
            name=c.class_name)
    
    Dataset = MVTecDataset if c.dataset == 'mvtec' else (VisADataset if c.dataset == 'visa' else MSTCDataset)

    train_dataset = Dataset(c, is_train=True)
    test_dataset  = Dataset(c, is_train=False)
    
    # Try to load pretrained velocity nets for this class
    vel_ckpt = os.path.join('work_dirs', 'velocity_preflow', c.dataset, c.class_name, 'velocity_last.pt')
    vel_model = None
    alpha_list = None
    if os.path.exists(vel_ckpt):
        try:
            # Build extractor to know stage channels
            extractor_tmp, c_list = build_extractor(c)
            from models.velocity import Velocity3Stage
            # Default per-stage cfg; overridden by checkpoint if present
            vel_cfg = {'s1_hidden':128,'s1_resblocks':2,'s2_hidden':128,'s2_resblocks':2,'s3_hidden':128,'s3_resblocks':2}
            vel_model = Velocity3Stage(c_list, vel_cfg).to(c.device)
            state = torch.load(vel_ckpt, map_location=c.device)
            if 'model' in state:
                vel_model.load_state_dict(state['model'])
            alpha_list = state.get('alpha', [1.0,1.0,1.0])
            vel_model.eval()
            print(f"Loaded velocity nets from {vel_ckpt}")
        except Exception as e:
            print(f"[WARN] Could not load velocity nets: {e}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [parallel_flow.to(c.device) for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.to(c.device)
    # if c.wandb_enable:
    #     for idx, parallel_flow in enumerate(parallel_flows):
    #         wandb.watch(parallel_flow, log='all', log_freq=100, idx=idx)
    #     wandb.watch(fusion_flow, log='all', log_freq=100, idx=len(parallel_flows))
    params = list(fusion_flow.parameters())
    for parallel_flow in parallel_flows:
        params += list(parallel_flow.parameters())
        
    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    start_epoch = 0
    if c.mode == 'test':
        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
        epoch = start_epoch + 1
        gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow, vel_model if 'vel_model' in locals() else None, alpha_list if 'alpha_list' in locals() else None)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        # New: process difference maps and sum for localization only
        _, anomaly_score_map_add_diff, _ = post_process(c, size_list, outputs_list_diff)
        anomaly_score_map_add = anomaly_score_map_add
        det_auroc, loc_auroc, loc_pro_auc, best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)
        
        return
    
    if c.resume:
        last_epoch = load_weights(parallel_flows, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=(c.lr_warmup_epochs - start_epoch)*c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None

    for epoch in range(start_epoch, c.meta_epochs):
        print()
        train_meta_epoch(c, epoch, train_loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None)

        gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow, vel_model if 'vel_model' in locals() else None, alpha_list if 'alpha_list' in locals() else None)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        anomaly_score_map_add = anomaly_score_map_add

        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        det_auroc, loc_auroc, loc_pro_auc, \
            best_det_auroc, best_loc_auroc, best_loc_pro = \
                eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, pro_eval)

        if c.wandb_enable:
            wandb_run.log(
                {
                    'Det.AUROC': det_auroc,
                    'Loc.AUROC': loc_auroc,
                    'Loc.PRO': loc_pro_auc
                },
                step=epoch
            )

        save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
        if best_loc_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        if best_loc_pro and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)
