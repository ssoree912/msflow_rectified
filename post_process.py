import datetime
import numpy as np
import torch
import torch.nn.functional as F

def post_process(c, size_list, outputs_list):
    print('Multi-scale sizes:', size_list)
    downscale = max(1, int(getattr(c, 'pixel_eval_downscale', 1)))
    if downscale > 1:
        target_h = max(1, int(c.input_size[0] // downscale))
        target_w = max(1, int(c.input_size[1] // downscale))
        target_size = (target_h, target_w)
    else:
        target_size = c.input_size
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]
    log_interval = max(1, int(getattr(c, "log_interval", 50)))
    for l, outputs in enumerate(outputs_list):
        if not outputs:
            logp_maps[l] = torch.empty((0, *target_size))
            prop_maps[l] = torch.empty((0, *target_size))
            continue
        total = sum(int(o.shape[0]) for o in outputs)
        processed = 0
        logp_chunks = []
        prop_chunks = []
        for i, out in enumerate(outputs):
            if out.dtype == torch.float16:
                out = out.float()
            logp = F.interpolate(out.unsqueeze(1),
                    size=target_size, mode='bilinear', align_corners=True).squeeze(1)
            output_norm = out - out.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            prob_map = torch.exp(output_norm) # convert to probs in range [0:1]
            prop = F.interpolate(prob_map.unsqueeze(1),
                    size=target_size, mode='bilinear', align_corners=True).squeeze(1)
            logp_chunks.append(logp)
            prop_chunks.append(prop)
            processed += int(out.shape[0])
            if log_interval and ((i == 0) or ((i + 1) % log_interval == 0) or processed == total):
                pct = 100.0 * processed / total if total else 100.0
                print(
                    datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
                    f"[PostProcess] level {l+1}/{len(outputs_list)} {processed}/{total} ({pct:.1f}%)"
                )
        logp_maps[l] = torch.cat(logp_chunks, 0)
        prop_maps[l] = torch.cat(prop_chunks, 0)
    
    logp_map = sum(logp_maps)
    logp_map-= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prop_map_mul = torch.exp(logp_map)
    anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
    batch = anomaly_score_map_mul.shape[0]
    top_k = int(target_size[0] * target_size[1] * c.top_k)
    if top_k < 1:
        top_k = 1
    anomaly_score = np.mean(
        anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)

    prop_map_add = sum(prop_maps)
    prop_map_add = prop_map_add.detach().cpu().numpy()
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()
