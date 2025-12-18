import torch
import torch.nn as nn
from models.openaimodel_medical import UNetModel as MedicalUNet

class StageVelocityUNet(nn.Module):
    def __init__(self, c_in: int, c_hidden: int = 128, num_res_blocks: int = 2, num_heads: int = 4):
        super().__init__()
        self.unet = MedicalUNet(
            image_size=None,
            in_channels=c_in,
            model_channels=c_hidden,
            out_channels=c_in,
            num_res_blocks=num_res_blocks,
            attention_resolutions=[],
            channel_mult=(1, 1, 2),
            conv_resample=True, dims=2, dropout=0.0,
            num_classes=None, use_checkpoint=False, use_fp16=False,
            num_heads=num_heads, num_head_channels=-1, num_heads_upsample=-1,
            use_scale_shift_norm=True, resblock_updown=False
        )

    def forward(self, x, t):
        """
        x: (B,C,H,W), t: (B,1) in [0,1]
        """
        return self.unet(x, t=t)  # pass t straight into MedicalUNet

class Velocity3Stage(nn.Module):
    def __init__(self, c_list, cfg):
        super().__init__()
        self.stage1 = StageVelocityUNet(c_list[0], cfg['s1_hidden'], cfg['s1_resblocks'], 4)
        self.stage2 = StageVelocityUNet(c_list[1], cfg['s2_hidden'], cfg['s2_resblocks'], 4)
        self.stage3 = StageVelocityUNet(c_list[2], cfg['s3_hidden'], cfg['s3_resblocks'], 4)

    def forward(self, xs, t):
        x1, x2, x3 = xs
        d1 = self.stage1(x1, t)
        d2 = self.stage2(x2, t)
        d3 = self.stage3(x3, t)
        return d1, d2, d3