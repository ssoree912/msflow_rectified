from typing import Optional, Tuple

import torch

from pruning.mnn_dwa import MaskConv2d


def resolve_pruning_mode(mode: Optional[str]) -> Tuple[str, Optional[str]]:
    mode = (mode or "dense").lower()
    legacy_modes = {"dense", "sparse", "static", "dynamic"}
    dwa_modes = {"reactivate_only", "kill_only", "kill_and_reactivate"}
    if mode in legacy_modes:
        return mode, None
    if mode in dwa_modes:
        return "sparse", mode
    raise ValueError(f"Unknown pruning_mode: {mode}")


def apply_pruning_mask(model, sparsity: float) -> float:
    if sparsity <= 0:
        for module in model.modules():
            if isinstance(module, MaskConv2d):
                module.mask.data.fill_(1.0)
        return 0.0

    weights = []
    modules = []
    for module in model.modules():
        if isinstance(module, MaskConv2d):
            weights.append(module.weight.data.abs().view(-1))
            modules.append(module)

    if not weights:
        return 0.0
    #가중치 절대값 모아서 threshold 계산. 전체 가중치 중 하위 몇 % 마스킹할지 결정
    all_weights = torch.cat(weights)
    if sparsity >= 1.0:
        threshold = all_weights.max()
    else:
        threshold = torch.quantile(all_weights, float(sparsity))

    total_params = 0
    pruned_params = 0
    for module in modules:
        w_abs = module.weight.data.abs()
        mask = (w_abs > threshold).float()
        module.mask.data = mask
        total_params += mask.numel()
        pruned_params += (mask == 0).sum().item()

    return pruned_params / max(1, total_params)


def extractor_forward(c, extractor, x):
    if hasattr(extractor, "set_control"):
        update_threshold = getattr(c, "dwa_update_threshold", False)
        out = extractor(
            x,
            type_value=getattr(c, "pruning_type_value", "sparse"),
            forward_type=getattr(c, "pruning_forward_type", None),
            alpha=getattr(c, "dwa_alpha", 1.0),
            beta=getattr(c, "dwa_beta", 1.0),
            update_threshold=update_threshold,
            threshold_percentile=getattr(c, "dwa_threshold_percentile", 50),
        )
        if update_threshold:
            setattr(c, "dwa_update_threshold", False)
        return out
    return extractor(x)
