from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmcv.runner import auto_fp16
from mmdet.models.necks import FPN

class FPNC(FPN):
    def __init__(self, conv_cfg=None, norm_cfg=None, act_cfg=None, final_dim=(900, 1600), downsample=4, 
                 use_adp=False, fuse_conv_cfg=None, outC=256, **kwargs):
        super(FPNC, self).__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        self.target_size = (final_dim[0] // downsample, final_dim[1] // downsample)
        self.use_adp = use_adp

        if self.use_adp:
            self.adp = nn.ModuleList()
            for i in range(self.num_outs):
                resize = nn.AdaptiveAvgPool2d(self.target_size) if i == 0 else nn.Upsample(
                    size=self.target_size, mode='bilinear', align_corners=True
                )
                adp = nn.Sequential(
                    resize,
                    ConvModule(self.out_channels, self.out_channels, 1, padding=0, 
                               conv_cfg=fuse_conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                               inplace=False)
                )
                self.adp.append(adp)
        
        self.reduc_conv = ConvModule(
            self.out_channels * self.num_outs, outC, 3, padding=1, conv_cfg=fuse_conv_cfg, 
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False
        )

    @auto_fp16()
    def forward(self, x):
        outs = super().forward(x)
        if len(outs) > 1:
            resize_outs = []
            if self.use_adp:
                for i in range(len(outs)):
                    feature = self.adp[i](outs[i])
                    resize_outs.append(feature)
            else:
                for i in range(len(outs)):
                    feature = outs[i]
                    if feature.shape[2:] != self.target_size:
                        feature = F.interpolate(feature, self.target_size,  mode='bilinear', align_corners=True)
                    resize_outs.append(feature)
            out = torch.cat(resize_outs, dim=1)
            out = self.reduc_conv(out)
        
        else:
            out = outs[0]
        return [out]

class GeneralizedLSSFPN(FPN):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, no_norm_on_lateral=False,
                 conv_cfg=None, norm_cfg=dict(type='BN2d'), act_cfg=dict(type='ReLU'), 
                 upsample_cfg=dict(mode="bilinear", align_corners=True), **kwargs):
        super().__init__(in_channels, out_channels, num_outs, start_level=start_level, end_level=end_level,
                         no_norm_on_lateral=no_norm_on_lateral, conv_cfg=conv_cfg, norm_cfg=norm_cfg, 
                         act_cfg=act_cfg, upsample_cfg=upsample_cfg, **kwargs)

        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels),
                out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg, inplace=False
            )
            self.lateral_convs.append(l_conv)
    @auto_fp16()   
    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals - 1)
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)
