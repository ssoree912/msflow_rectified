import os
import time
import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from datasets import MVTecDataset, VisADataset, MSTCDataset, MSTCFeatureDataset
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from models.velocity import Velocity3Stage
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights, infer_stage_hw
from evaluations import eval_det_loc
from pruning.utils import resolve_pruning_mode, apply_pruning_mask, extractor_forward

#한 배치를 flow까지 통과시키는 학숨
def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    h_list = extractor_forward(c, extractor, image) #고정으로 쓰는 백본, 특징 추출기
    #논문에서 말하는 fearure pyramid를 pooling으로 정리 , stage 마다 조건으로 encoding을 넣고 parallel flow에 전달
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

    z_list, fuse_jac = fusion_flow(z_list) #stage 별 z들을  fusion flow가 스케일간 상호작용
    jac = fuse_jac + sum(parallel_jac_list) #전체 자코비안은 각 스테이지의 자코비안과 퓨전 플로우의 자코비안을 더한 것

    return z_list, jac


def model_forward_features(c, parallel_flows, fusion_flow, feats_list):
    # feats_list: list of pooled feature maps per stage (B,C,H,W)
    z_list = []
    parallel_jac_list = []
    for idx, (y, parallel_flow, c_cond) in enumerate(zip(feats_list, parallel_flows, c.c_conds)):
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
        for idx, batch in enumerate(loader):
            optimizer.zero_grad()
            if c.feature_cache:
                f1, f2, f3, _, _ = batch
                feats = [f1.to(c.device), f2.to(c.device), f3.to(c.device)]
            else:
                image, _, _ = batch
                image = image.to(c.device)
            if scaler:
                with autocast():
                    if c.feature_cache:
                        z_list, jac = model_forward_features(c, parallel_flows, fusion_flow, feats)
                    else:
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
                if c.feature_cache:
                    z_list, jac = model_forward_features(c, parallel_flows, fusion_flow, feats)
                else:
                    z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                loss = 0.
                for z in z_list:
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3)) #NLL loss : 0.5 * ||z||^2, 정상 데이터를 높은 loglikelihood로 만들도록 학습하는 구조
                loss = loss - jac
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2)
                optimizer.step()
            epoch_loss += t2np(loss)
            if c.feature_cache:
                image_count += feats[0].shape[0]
            else:
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
        for idx, batch in enumerate(loader):
            if c.feature_cache:
                f1, f2, f3, label, mask = batch
                feats = [f1.to(c.device), f2.to(c.device), f3.to(c.device)]
            else:
                image, label, mask = batch
                image = image.to(c.device)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            if c.feature_cache:
                z_list, jac = model_forward_features(c, parallel_flows, fusion_flow, feats)
            else:
                z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                # Original logp map,  stage 별 logp map 저장
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
            if c.feature_cache:
                image_count += feats[0].shape[0]
            else:
                image_count += image.shape[0]

        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                epoch, mean_epoch_loss, fps))

    return gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff



def train(c):
    pruning_mode = getattr(c, 'pruning_mode', 'dense')
    pruning_type_value, pruning_forward_type = resolve_pruning_mode(pruning_mode)
    c.pruning_mode = pruning_mode
    c.pruning_type_value = pruning_type_value
    c.pruning_forward_type = pruning_forward_type
    c.pruning_sparsity = getattr(c, 'pruning_sparsity', 0.0)
    c.dwa_alpha = getattr(c, 'dwa_alpha', 1.0)
    c.dwa_beta = getattr(c, 'dwa_beta', 1.0)
    c.dwa_update_threshold = getattr(c, 'dwa_update_threshold', False)
    c.dwa_threshold_percentile = getattr(c, 'dwa_threshold_percentile', 50)
    
    if c.wandb_enable:
        wandb.finish()
        wandb_run = wandb.init(
            project='65001-msflow', 
            group=c.version_name,
            name=c.class_name)
    
    if c.feature_cache:
        if c.dataset != 'mstc':
            raise ValueError("feature cache is only supported for mstc")
        Dataset = MSTCFeatureDataset
    else:
        Dataset = MVTecDataset if c.dataset == 'mvtec' else (VisADataset if c.dataset == 'visa' else MSTCDataset)

    train_dataset = Dataset(c, is_train=True)
    test_dataset = Dataset(c, is_train=False)
    
    # Try to load pretrained velocity nets for this class
    vel_ckpt = os.path.join('work_dirs', 'velocity_preflow', c.dataset, c.class_name, 'velocity_last.pt')
    vel_model = None
    alpha_list = None
    if not c.feature_cache and os.path.exists(vel_ckpt):
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

    if c.feature_cache:
        extractor = None
        output_channels = [int(x) for x in train_dataset.output_channels]
        stage_hw = [(int(h), int(w)) for h, w in train_dataset.stage_hw]
        print(f"[FeatureCache] Using cached features from {c.feature_subdir}, skip extractor.")
    else:
        extractor, output_channels = build_extractor(c)
        if c.pruning_type_value == "dense":
            applied_sparsity = apply_pruning_mask(extractor, 0.0)
        else:
            applied_sparsity = apply_pruning_mask(extractor, c.pruning_sparsity)
        c.pruning_applied_sparsity = applied_sparsity
        print(
            f"[Pruning] mode={c.pruning_mode} "
            f"(type_value={c.pruning_type_value}, forward_type={c.pruning_forward_type}), "
            f"sparsity={c.pruning_sparsity:.4f}, applied={applied_sparsity:.4f}"
        )
        stage_hw = infer_stage_hw(c, extractor)
        extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels, stage_hw)
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
    last_fps = None
    best_det_fps = None
    best_loc_fps = None

    start_epoch = 0
    if c.mode == 'test':
        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
        epoch = start_epoch + 1
        eval_start = time.time()
        gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow, vel_model if 'vel_model' in locals() else None, alpha_list if 'alpha_list' in locals() else None)
        last_fps = len(test_loader.dataset) / (time.time() - eval_start)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        # New: process difference maps and sum for localization only
        _, anomaly_score_map_add_diff, _ = post_process(c, size_list, outputs_list_diff)
        anomaly_score_map_add = anomaly_score_map_add
        pixel_eval = getattr(c, 'pixel_eval', True)
        det_auroc, loc_auroc, loc_pro_auc, best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(
            det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch,
            gt_label_list, anomaly_score, gt_mask_list,
            anomaly_score_map_add, anomaly_score_map_mul,
            c.pro_eval, pixel_eval=pixel_eval
        )
        
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

        eval_start = time.time()
        gt_label_list, gt_mask_list, outputs_list, size_list, outputs_list_diff = inference_meta_epoch(c, epoch, test_loader, extractor, parallel_flows, fusion_flow, vel_model if 'vel_model' in locals() else None, alpha_list if 'alpha_list' in locals() else None)
        last_fps = len(test_loader.dataset) / (time.time() - eval_start)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)
        anomaly_score_map_add = anomaly_score_map_add

        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        pixel_eval = getattr(c, 'pixel_eval', True)
        det_auroc, loc_auroc, loc_pro_auc, \
            best_det_auroc, best_loc_auroc, best_loc_pro = \
                eval_det_loc(
                    det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch,
                    gt_label_list, anomaly_score, gt_mask_list,
                    anomaly_score_map_add, anomaly_score_map_mul,
                    pro_eval, pixel_eval=pixel_eval
                )
        if best_det_auroc:
            best_det_fps = last_fps
        if best_loc_auroc:
            best_loc_fps = last_fps

        if c.wandb_enable:
            log_payload = {'Det.AUROC': det_auroc}
            if loc_auroc is not None:
                log_payload['Loc.AUROC'] = loc_auroc
            if loc_pro_auc is not None:
                log_payload['Loc.PRO'] = loc_pro_auc
            wandb_run.log(log_payload, step=epoch)

        save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
        if best_loc_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        if best_loc_pro and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)

    if c.mode == 'train' and last_fps is not None:
        if best_det_fps is None:
            best_det_fps = last_fps
        pixel_eval = getattr(c, 'pixel_eval', True)
        if pixel_eval and best_loc_fps is None:
            best_loc_fps = last_fps
        out_path = os.path.join(c.ckpt_dir, 'best_metrics.csv')
        with open(out_path, 'w') as f:
            f.write('dataset,class_name,best_det_auroc,best_det_epoch,best_det_fps,best_loc_auroc,best_loc_epoch,best_loc_fps\n')
            if pixel_eval:
                loc_score = f'{loc_auroc_obs.max_score:.4f}'
                loc_epoch = f'{loc_auroc_obs.max_epoch}'
                loc_fps = f'{best_loc_fps:.4f}'
            else:
                loc_score = 'na'
                loc_epoch = 'na'
                loc_fps = 'na'
            f.write(
                f'{c.dataset},{c.class_name},'
                f'{det_auroc_obs.max_score:.4f},{det_auroc_obs.max_epoch},'
                f'{best_det_fps:.4f},'
                f'{loc_score},{loc_epoch},'
                f'{loc_fps}\n'
            )
        print(f"Saved best metrics to {out_path}")
