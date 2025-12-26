import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


################################################################
# 1) 기존 Static / Dynamic (DPF) 마스커
################################################################

class MaskerStatic(torch.autograd.Function):
    """Static pruning: dead weights stay dead (gradient masked by mask)"""
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        (mask,) = ctx.saved_tensors
        return grad_out * mask, None


class MaskerDynamic(torch.autograd.Function):
    """DPF: dead weights can reactivate (no gradient masking)"""
    @staticmethod
    def forward(ctx, x, mask):
        # dynamic: no need to save mask for backward since grad is unchanged
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


################################################################
# 2) DWA (Dynamic Weight Adjustment) – 3가지 실험용 마스커
################################################################

class MaskerScalingReactivationOnly(torch.autograd.Function):
    """
    (1) Reactivation-only:
        g' = g*m + alpha * g * (1-m) * f
      where f = ||w| - tau|
    """
    @staticmethod
    def forward(ctx, x, mask, alpha, threshold):
        ctx.save_for_backward(mask, x, threshold)
        ctx.alpha = alpha
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x, threshold = ctx.saved_tensors
        alpha = ctx.alpha
        f = torch.abs(torch.abs(x) - threshold)  # f = ||w| - tau|
        g_new = (grad_out * mask) + (grad_out * (1 - mask) * f * alpha)
        return g_new, None, None, None


class MaskerScalingKillOnly(torch.autograd.Function):
    """
    (2) Kill Only (활성 가중치만 죽임):
        g' = beta * g * m * |w| + g * (1-m)
      활성 가중치는 |w|로 스케일해 '죽이는' 방향으로, 비활성은 평범한 grad (재활성화 효과 없음)
    """
    @staticmethod
    def forward(ctx, x, mask, beta):
        ctx.save_for_backward(mask, x)
        ctx.beta = beta
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x = ctx.saved_tensors
        beta = ctx.beta
        b = torch.abs(x)  # |w|
        g_new = (grad_out * mask * b * beta) + (grad_out * (1 - mask))
        return g_new, None, None


class MaskerScalingKillAndReactivate(torch.autograd.Function):
    """
    (3) Kill & Reactivate (양쪽 모두):
        g' = beta * g * m * |w| + alpha * g * (1-m) * f
      where f = ||w| - tau|
    """
    @staticmethod
    def forward(ctx, x, mask, alpha, beta, threshold):
        ctx.save_for_backward(mask, x, threshold)
        ctx.alpha = alpha
        ctx.beta = beta
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x, threshold = ctx.saved_tensors
        alpha, beta = ctx.alpha, ctx.beta
        f = torch.abs(torch.abs(x) - threshold)  # f = ||w| - tau|
        b = torch.abs(x)  # |w|
        g_new = (grad_out * mask * b * beta) + (grad_out * (1 - mask) * f * alpha)
        return g_new, None, None, None, None


################################################################
# 3) 통합 MaskConv2d
#    - legacy: type_value 사용 (string)
#    - DWA   : forward_type 사용 (string)
################################################################

class MaskConv2d(nn.Conv2d):
    """
    통합 MaskConv2d
    - legacy 모드 (기본): self.forward_type is None -> self.type_value에 따라 동작
        * "sparse" : weight * mask (출력만 sparse)
        * "static" : MaskerStatic (gradient도 mask로 차단)
        * "dynamic": MaskerDynamic (gradient 그대로 통과)
        * "dense"  : weight (마스크 미적용)
    - DWA 모드: self.forward_type in {"reactivate_only", "kill_only", "kill_and_reactivate"}
        * 아래 3가지 마스커를 사용
    """
    _DWA_MODES = {"reactivate_only", "kill_only", "kill_and_reactivate"}
    _LEGACY_MODES = {"sparse", "static", "dynamic", "dense"}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        # 공통: pruning mask (학습 X)
        self.mask = Parameter(torch.ones_like(self.weight), requires_grad=False)

        # legacy 경로
        self.type_value = "sparse"

        # DWA 경로
        self.forward_type = None  # None => legacy, 문자열 모드면 DWA 사용
        self.alpha = 1.0
        self.beta = 1.0
        self.threshold = Parameter(torch.tensor(0.05, dtype=self.weight.dtype, device=self.weight.device),
                                   requires_grad=False)

    # 편의 메서드: DWA threshold 업데이트(가중치 절대값의 p-분위수)
    def update_threshold(self, percentile: int = 50):
        with torch.no_grad():
            weight_abs = torch.abs(self.weight)
            self.threshold.data = torch.quantile(weight_abs, percentile / 100.0)

    def _normalize_forward_type(self):
        if self.forward_type is None:
            return None
        if isinstance(self.forward_type, str):
            ft = self.forward_type.lower()
            if ft in self._DWA_MODES:
                return ft
            raise NotImplementedError(f"Unknown forward_type: {self.forward_type}")
        raise TypeError(f"forward_type must be str/None, got {type(self.forward_type)}")

    def _normalize_legacy_mode(self):
        if isinstance(self.type_value, str):
            key = self.type_value.lower()
            if key in self._LEGACY_MODES:
                return key
            raise NotImplementedError(f"Unknown type_value: {self.type_value}")
        raise TypeError(f"type_value must be str, got {type(self.type_value)}")

    def _apply_dwa(self, forward_type):
        if forward_type == "reactivate_only":
            return MaskerScalingReactivationOnly.apply(
                self.weight, self.mask, self.alpha, self.threshold
            )
        if forward_type == "kill_only":
            return MaskerScalingKillOnly.apply(self.weight, self.mask, self.beta)
        if forward_type == "kill_and_reactivate":
            return MaskerScalingKillAndReactivate.apply(
                self.weight, self.mask, self.alpha, self.beta, self.threshold
            )
        raise NotImplementedError(f"Unknown forward_type: {self.forward_type}")

    def _apply_legacy(self):
        mode = self._normalize_legacy_mode()
        if mode == "static":
            return MaskerStatic.apply(self.weight, self.mask)
        if mode == "dynamic":
            return MaskerDynamic.apply(self.weight, self.mask)
        if mode == "dense":
            return self.weight
        # Default: sparse output only
        return self.weight * self.mask

    def forward(self, x):
        # 1) DWA 모드가 설정되어 있으면 우선 적용
        forward_type = self._normalize_forward_type()
        if forward_type is not None:
            masked_weight = self._apply_dwa(forward_type)
        # 2) legacy 경로(type_value)
        else:
            masked_weight = self._apply_legacy()

        # 표준 Conv
        return F.conv2d(x, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
