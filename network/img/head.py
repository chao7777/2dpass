import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def sigmoid_focal_loss(input, target, alpha=-1, gamma=2, reduction='mean'):
    input, target = input.float(), target.float()
    p = torch.sigmoid(input)
    ce_loss = F.binary_cross_entropy_with_logits(input, target, reduce='none')
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha > 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

class BEVGridTransform(nn.Module):
    def __init__(self, input_scope, output_scope, prescale_factor) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x):
        if self.prescale_factor != 1:
            x = F.interpolate(x, scale_factor=self.prescale_factor, mode='bilinear', align_corners=False)
        coods = []
        for (imin, imax, _), (omin, omax, ostep) in zip(self.input_scope, self.output_scope):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 -1
            coods.append(v.to(x.device))
        u, v = torch.meshgrid(coods, indexing='ij')
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)
        x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        return x


class SegHead(nn.Module):
    """bevfusion_mit"""
    def __init__(self, in_channels, grid_transform, classes, loss) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss

        self.bev_grid_transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1)
        )
    
    def forward(self, x, target):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.bev_grid_transform(x)
        x = self.classifier(x)
        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == 'xent':
                    loss = F.binary_cross_entropy_with_logits(x[:, index].float(), target[:, index].float(), reduce='mean')
                elif self.loss == 'local':
                    loss = sigmoid_focal_loss(x[:, index], target[:, index])
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}/{self.loss}"] = loss
            return losses
        else:
            return torch.sigmoid(x)
