import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map

from detectron2.layers import Conv2d, ShapeSpec, get_norm

__all__ = ["build_resnet_fan_backbone", "build_retinanet_resnet_fan_backbone", "FAN"]

# from dcn_v2 import DCN, DCNPooling, DCNv2, DCNv2Pooling, dcn_v2_conv, dcn_v2_pooling
from dcn_v2 import DCN as dcn_v2
from detectron2.layers import (CNNBlockBase, Conv2d, DeformConv, ModulatedDeformConv, ShapeSpec, get_norm, )


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return feat_align + feat_arm


class FAN(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, strides, in_channels_per_feature, out_channels, norm=""):
        super(FAN, self).__init__()

        align_modules = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature[:-1]):
            stage = int(math.log2(strides[idx]))
            lateral_norm = get_norm(norm, out_channels)
            align_module = FeatureAlign_V2(in_channels, out_channels, norm=lateral_norm)  # proposed fapn
            self.add_module("fan_align{}".format(stage), align_module)
            align_modules.append(align_module)
            output_conv = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                 norm=get_norm(norm, out_channels), )
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)
            output_convs.append(output_conv)
        stage = int(math.log2(strides[len(in_channels_per_feature) - 1]))
        lateral_conv = Conv2d(in_channels_per_feature[-1], out_channels, kernel_size=1, bias=use_bias,
                              norm=get_norm(norm, out_channels))
        align_modules.append(lateral_conv)
        self.add_module("fan_align{}".format(stage), lateral_conv)
        # Place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.
        self.align_modules = align_modules[::-1]
        self.output_convs = output_convs[::-1]
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}

    def forward(self, bottom_up_features):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Feature maps into top-down order (from low to high resolution)
        x = bottom_up_features[::-1]
        results = []
        prev_features = self.align_modules[0](x[0])
        results.append(prev_features)
        for features, align_module, output_conv in zip(x[1:], self.align_modules[1:], self.output_convs[0:]):
            prev_features = align_module(features, prev_features)
            results.insert(0, output_conv(prev_features))

        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))
