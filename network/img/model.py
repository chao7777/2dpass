import torch
from torch import nn as nn
from torch.nn import functional as F

from network.img.swin_transform import SwinTransformer
from network.img.lss import LiftSplatShoot
from network.img.fpn import GeneralizedLSSFPN

from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer

class SE(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid) -> None:
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act_layer = act_layer
        self.conv_expand = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gate = gate_layer

    def forward(self, x, x_se):
        x_se = self.conv_expand(self.act_layer(self.conv_reduce(x_se)))
        return x * x_se


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, batch_norm) -> None:
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = batch_norm(planes)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channel=256, batch_norm=nn.BatchNorm2d, dilations=[1, 6, 12, 18]) -> None:
        super().__init__()
        self.aspp1 = ASPPModule(inplanes, mid_channel, 1, 0, dilations[0], batch_norm)
        self.aspp2 = ASPPModule(inplanes, mid_channel, 3, dilations[1], dilations[1], batch_norm)
        self.aspp3 = ASPPModule(inplanes, mid_channel, 3, dilations[2], dilations[2], batch_norm)
        self.aspp4 = ASPPModule(inplanes, mid_channel, 3, dilations[3], dilations[3], batch_norm)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channel, 1, stride=1, bias=False),
            batch_norm(mid_channel),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(int(mid_channel * 5), mid_channel, 1, bias=False)
        self.bn1 = batch_norm(mid_channel)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.init_weight()
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.drop_out(x)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels) -> None:
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = MLP(27, mid_channels, mid_channels)
        self.depth_se = SE(mid_channels)
        self.context_mlp = MLP(27, mid_channels, mid_channels)
        self.context_se = SE(mid_channels)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x, data_dict):
        #bevdepth处理data_dict，获取mlp_input，
        mlp_input = None
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class ImgDetector(nn.Module):
    """bevdepth"""
    def __init__(self, final_dim, out_channels, img_config) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.img_backbone = SwinTransformer(**img_config['img_backbone_conf'])
        self.img_neck = GeneralizedLSSFPN(**img_config['img_neck_conf'])
        self.lss = LiftSplatShoot(camera_depth_range=[2.0, 58.0, 0.5], pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3], downsample=16)
        self.img_backbone.init_weights()
        self.img_neck.init_weights()
        
        self.depth_channels, _, _, _ = self.lss.frustum.shape
        self.voxel_size, self.voxel_coord, self.voxel_num = self.lss.dx, self.lss.bx, self.lss.nx
        self.depth_net = self.config_depth_net(img_config['depth_net_conf'])
        

    def config_depth_net(self, depth_net_conf):
        return DepthNet(depth_net_conf['in_channels'], depth_net_conf['mid_channels'], self.out_channels, self.depth_channels)

    def forward(self, sweep_img, data_dict, is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps, num_cameras, 3, H, W).
            data_dict(dict):data info
            timestamps(Tensor): Timestamp for all images with the shape of(B, num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        _, num_sweeps, _, _, _, _ = sweep_img.shape
        feature_map, depth_feature = self.forward_single_sweep(0, sweep_img[:, 0:1, ...], data_dict)
        if num_sweeps == 1:
            return feature_map
        reature_list = [feature_map]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map, _ = self.forward_single_sweep(sweep_index, sweep_img[:, sweep_index:sweep_index+1, ...], data_dict)
                reature_list.append(feature_map)
        return torch.cat(reature_list, 1), depth_feature
        

    def forward_single_sweep(self, sweep_index, sweep_imgs, data_dict):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            data_dict (dict):data info
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, _, num_cams, _, _, _ = sweep_imgs.shape
        imgs_feats = self.get_cam_feats(sweep_imgs)
        source_feature = imgs_feats[:, 0, ...]
        C, H, W = source_feature.shape[2:4]
        depth_feature = self.depth_net(source_feature.reshape(batch_size * num_cams, C, H, W), data_dict)
        depth = depth_feature[:, depth_feature].softmax(dim=1, dtype=depth_feature.dtype)
        geom_xyz = self.get_geometry(data_dict)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
        img_feat_with_depth = depth.unsqueeze(1) * depth_feature[:, self.depth_channels:(self.depth_channels + self.out_channels)].unsqueeze(2)
        cam_channel, frustum_d, H, W = img_feat_with_depth.shape[1:4]
        img_feat_with_depth = img_feat_with_depth.reshape(batch_size, num_cams, cam_channel, frustum_d, H, W)
        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
        feature_map = self.voxel_pooling(geom_xyz)
        return feature_map.contiguous(), depth_feature[:, self.depth_channels].softmax(dim=1)

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape
        imgs = imgs.flatten().view(batch_size * num_sweeps* num_cams, num_channels, imH, imW)
        imgs_feats = self.img_neck(self.img_backbone(imgs))
        C, H, W = imgs_feats.shape[1:3]
        imgs_feats = imgs_feats.reshape(batch_size, num_sweeps, num_cams, C, H, W)
        return imgs_feats
    
    def get_geometry(self, data_dict):
        """Transfer points from camera coord to ego coord."""
        pass

    def voxel_pooling(self, geom_feat):
        """Forward function for `voxel pooling, return bev feature map"""
        pass