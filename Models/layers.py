import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# from dcn_v2 import DCN as dcn_v2


###########################################################################
########################## LAYER BLOCKS ###################################
###########################################################################


def Norm2d(planes):
    return nn.BatchNorm2d(planes)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class LinearReLuBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearReLuBn, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class PixelShuffle_ICNR(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, blur=False):
        super(PixelShuffle_ICNR, self).__init__()
        self.blur = blur
        self.conv = nn.Conv2d(in_channels, out_channels * (scale**2), kernel_size=1, padding=0)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur_kernel = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur_kernel(self.pad(x)) if self.blur else x


###########################################################################
########################### CONV BLOCKS ###################################
###########################################################################


class ConvBn2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=True,
        padding=(1, 1),
        groups=1,
        dilation=1,
        relu=False,
        bn=False,
    ):
        super(ConvBn2d, self).__init__()
        self.relu = relu
        self.bn = bn
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
            dilation=dilation,
        )
        if self.bn:
            self.bn = Norm2d(out_channels)
        if self.relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class LargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(LargeKernelConv, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv1_1 = ConvBn2d(
            in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, pad)
        )
        self.conv1_2 = ConvBn2d(
            out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0)
        )

        self.conv2_1 = ConvBn2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0)
        )
        self.conv2_2 = ConvBn2d(
            out_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, pad)
        )

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)

        return x1 + x2


###########################################################################
####################### SQUEEZE EXCITATION BLOCKS #########################
###########################################################################


class SpatialGate2d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialGate2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.conv1(x)
        cal = self.sigmoid(cal)
        return cal * x


class ChannelGate2d(nn.Module):
    def __init__(self, channels, reduction=2):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.avg_pool(x)
        cal = self.fc1(cal)
        cal = self.relu(cal)
        cal = self.fc2(cal)
        cal = self.sigmoid(cal)

        return cal * x


class scSqueezeExcitationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super(scSqueezeExcitationGate, self).__init__()
        self.spatial_gate = SpatialGate2d(channels)
        self.channel_gate = ChannelGate2d(channels, reduction=reduction)

    def forward(self, x, z=None):
        XsSE = self.spatial_gate(x)
        XcSe = self.channel_gate(x)
        return XsSE + XcSe


###########################################################################
########################## PYRAMID POOLING BLOCKS #########################
###########################################################################


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AASPP(nn.Module):
    def __init__(
        self, in_channels, channels, out_channels, output_stride, assimetry, attention=False
    ):
        super().__init__()
        self.attention = attention
        if output_stride == 16:
            dilations = [(d * assimetry[0], d * assimetry[1]) for d in [1, 3, 6, 9]]
        elif output_stride == 8:
            dilations = [(d * assimetry[0], d * assimetry[1]) for d in [1, 6, 12, 18]]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(
            in_channels, channels, 1, padding=0, dilation=dilations[0], BatchNorm=nn.BatchNorm2d
        )
        self.aspp2 = _ASPPModule(
            in_channels,
            channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=nn.BatchNorm2d,
        )
        self.aspp3 = _ASPPModule(
            in_channels,
            channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=nn.BatchNorm2d,
        )
        self.aspp4 = _ASPPModule(
            in_channels,
            channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=nn.BatchNorm2d,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(int(channels * 5), out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        if self.attention:
            self.att = EfficientChannelAttention(out_channels)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        if self.attention:
            x = self.att(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == "drn":
            inplanes = 512
        elif backbone == "mobilenet":
            inplanes = 320
        elif backbone == "r34":
            inplanes = 512
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(
            inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm
        )
        self.aspp2 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm
        )
        self.aspp3 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm
        )
        self.aspp4 = _ASPPModule(
            inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabDecoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super().__init__()
        if backbone == "resnet" or backbone == "drn":
            low_level_inplanes = 256
        elif backbone == "r34":
            low_level_inplanes = 64
        elif backbone == "xception":
            low_level_inplanes = 128
        elif backbone == "mobilenet":
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.logits = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        x = self.logits(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


###########################################################################
########################### ATTENTION BLOCKS ##############################
###########################################################################


class AttentionGate(nn.Module):
    def __init__(self, in_skip, in_g):
        super(AttentionGate, self).__init__()
        inter_c = in_skip // 2
        self.conv_skip = nn.Conv2d(
            in_skip, inter_c, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.conv_g = nn.Conv2d(in_g, inter_c, kernel_size=1, padding=0, bias=True)
        self.conv_psi = nn.Conv2d(inter_c, 1, kernel_size=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_skip, in_skip, kernel_size=1, padding=0), Norm2d(in_skip)
        )
        self.relu = nn.ReLU(inplace=True)
        # Initialise weights
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, x, g):
        theta_x = self.conv_skip(x)
        phi_g = self.conv_g(g)
        i = self.relu(theta_x + phi_g)
        i = self.conv_psi(i)
        i = torch.sigmoid(i)
        i = F.upsample(i, scale_factor=2, mode="bilinear", align_corners=False)
        i = i.expand_as(x) * x
        return self.W(i)


###########################################################################
############################ PANet BLOCKS #################################
###########################################################################


class FeaturePyramidAttention(nn.Module):
    def __init__(self, channels, out_channels=None):
        super(FeaturePyramidAttention, self).__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1p = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv3a = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv5a = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv5b = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.conv7a = nn.Conv2d(channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.conv7b = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, mode="std"):
        H, W = x.shape[2:]
        # down-path
        if mode == "std":
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == "reduced":
            xup1 = self.conv7a(x)
        elif mode == "extended":
            xup1 = F.avg_pool2d(x, kernel_size=4, stride=4)
            xup1 = self.conv7a(xup1)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        xup2 = self.upsample(xup3) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode="bilinear", align_corners=True)

        # Merge
        if mode == "std":
            x1 = self.upsample(xup1) * x1
        elif mode == "reduced":
            x1 = xup1 * x1
        elif mode == "extended":
            x1 = F.upsample(xup1, scale_factor=4, mode="bilinear", align_corners=True) * x1
        x1 = x1 + gp
        return x1


class FeaturePyramidAttention_v2(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)
        self.conv1p = ConvBn2d(
            channels, out_channels, kernel_size=1, stride=1, padding=0, act=True
        )

        self.conv3a = ConvBn2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True
        )
        self.conv3b = ConvBn2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True
        )

        self.conv5a = ConvBn2d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True
        )
        self.conv5b = ConvBn2d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True
        )

        self.conv7a = ConvBn2d(
            channels, out_channels, kernel_size=7, stride=1, padding=3, act=True
        )
        self.conv7b = ConvBn2d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, act=True
        )

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, mode="std"):
        H, W = x.shape[2:]
        # down-path
        if mode == "std":
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == "reduced":
            xup1 = self.conv7a(x)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        xup2 = self.upsample(xup3) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode="bilinear", align_corners=True)

        # Merge
        if mode == "std":
            x1 = self.upsample(xup1) * x1
        elif mode == "reduced":
            x1 = xup1 * x1
        x1 = x1 + gp
        return x1


class SmallFeaturePyramidAttention(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)
        self.conv1p = ConvBn2d(
            channels, out_channels, kernel_size=1, stride=1, padding=0, act=True
        )

        self.conv3a = ConvBn2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True
        )
        self.conv3b = ConvBn2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True
        )

        self.conv5a = ConvBn2d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True
        )
        self.conv5b = ConvBn2d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True
        )

        self.conv7a = ConvBn2d(
            channels, out_channels, kernel_size=7, stride=1, padding=3, act=True
        )
        self.conv7b = ConvBn2d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, act=True
        )

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        H, W = x.shape[2:]
        # down-path
        xup1 = self.conv3a(x)
        xup2 = self.conv5a(xup1)
        xup3 = self.conv7a(xup2)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv3b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv7b(xup3)

        # up-path
        xup2 = xup3 + xup2
        xup1 = xup2 + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = F.upsample(gp, size=(H, W), mode="bilinear", align_corners=True)

        x1 = xup1 * x1
        x1 = x1 + gp
        return x1


class GlobalAttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, out_channels=None):
        super(GlobalAttentionUpsample, self).__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)
        # GlobalPool and conv1
        cal1 = self.GPool(x)
        cal1 = self.conv1(cal1)
        cal1 = self.relu(cal1)

        # Calibrate skip connection
        skip = cal1 * skip
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x


class AttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, n_classes, out_channels=None):
        super().__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.rconv1 = nn.Conv2d(channels, n_classes, kernel_size=1, padding=0)
        self.gconv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)

        # GlobalPool and conv1
        gcalib = self.GPool(x)
        gcalib = self.gconv1(gcalib)
        gcalib = torch.sigmoid(gcalib)

        # RegionalPool
        rcalib = self.rconv1(x)
        rcalib = torch.sigmoid(rcalib)

        # Calibrate skip connection
        skip = (gcalib * skip) + (rcalib * skip)
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x, rcalib


###########################################################################
########################### DECODER BLOCKS ################################
###########################################################################


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        convT_channels=0,
        convT_ratio=0,
        SE=False,
        residual=False,
    ):
        super(UNetDecoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.SE = SE
        self.residual = residual
        self.convT_ratio = convT_ratio

        self.conv1 = ConvBn2d(
            in_channels, channels, kernel_size=3, padding=1, relu=False, bn=False
        )
        self.conv2 = ConvBn2d(
            channels, out_channels, kernel_size=3, padding=1, relu=False, bn=False
        )

        if convT_ratio:
            self.convT = nn.ConvTranspose2d(
                convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2
            )
            if residual:
                self.conv_res = nn.Conv2d(
                    convT_channels // convT_ratio, out_channels, kernel_size=1, padding=0
                )
        else:
            if residual:
                self.conv_res = nn.Conv2d(convT_channels, out_channels, kernel_size=1, padding=0)

        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, skip=None):
        if self.convT_ratio:
            x = self.convT(x)
            x = self.activation(x)
        x = self.upsample(x)

        residual = x
        if skip is not None:
            x = torch.cat([x, skip], 1)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.SE:
            x = self.scSE(x)

        if self.residual:
            x += self.conv_res(residual)
        x = self.activation(x)
        return x


class PSDecoder(nn.Module):
    def __init__(self, in_channels, skip_chanels, channels, out_channels, PS=True, SE=False):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)

        if PS:
            self.upsample = nn.PixelShuffle(upscale_factor=2)
            factor = 4
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            factor = 1

        self.SE = SE

        self.conv1 = ConvBn2d(
            in_channels // factor + skip_chanels, channels, kernel_size=3, padding=1
        )
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels // factor, out_channels, kernel_size=1, padding=0)

        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)

        residual = x
        if skip is not None:
            x = torch.cat([x, skip], 1)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.SE:
            x = self.scSE(x)

        x += self.conv_res(residual)
        x = self.activation(x)
        return x


class AdjascentPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad2d(padding=1)
        self._range = range(3)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_x = self.pad(x)
        out = []
        for i in self._range:
            for j in self._range:
                out.append(pad_x[:, :, i : i + H, j : j + W])
        out = torch.cat(out, dim=1).mean(dim=1, keepdim=True)
        return out


###########################################################################
############################## SCNN BLOCKS ################################
###########################################################################


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = Norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = Norm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class GatedConv2d(nn.Module):
    def __init__(self, s_in, r_in):
        super().__init__()
        self.conv1_att = nn.Sequential(
            nn.Conv2d(r_in + s_in, s_in, kernel_size=1), Norm2d(s_in), nn.Sigmoid()
        )
        self.conv1_weight = nn.Conv2d(s_in, s_in, kernel_size=1)

    def forward(self, s, r):
        # Compute alpha
        alpha = torch.cat([s, r], dim=1)
        alpha = self.conv1_att(alpha)

        # Apply residual and attention
        s = (s * alpha) + s

        # Final conv
        return self.conv1_weight(s)


class ShapeStream(nn.Module):
    def __init__(self, l1_in, l3_in, l4_in, l5_in):
        super().__init__()
        self.conv1_l1 = ConvBn2d(l1_in, l1_in, kernel_size=1, padding=0, act=True)
        self.conv1_l3 = ConvBn2d(l3_in, l1_in, kernel_size=1, padding=0, act=True)
        self.conv1_l4 = ConvBn2d(l4_in, l1_in, kernel_size=1, padding=0, act=True)
        self.conv1_l5 = ConvBn2d(l5_in, l1_in, kernel_size=1, padding=0, act=True)

        self.res_block1 = ResidualBlock(l1_in, l1_in)
        self.res_block2 = ResidualBlock(l1_in, l1_in)
        self.res_block3 = ResidualBlock(l1_in, l1_in)

        self.gconv1 = GatedConv2d(l1_in, l1_in)
        self.gconv2 = GatedConv2d(l1_in, l1_in)
        self.gconv3 = GatedConv2d(l1_in, l1_in)

        self.conv1_out = ConvBn2d(l1_in, 1, kernel_size=1, padding=0)
        self.conv1_grad_out = ConvBn2d(2, 1, kernel_size=1, padding=0)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, l1, l3, l4, l5, grad=None):
        x = self.conv1_l1(l1)
        x = self.res_block1(x)
        x = self.gconv1(x, self.up2(self.conv1_l3(l3)))

        x = self.res_block2(x)
        x = self.gconv2(x, self.up4(self.conv1_l4(l4)))

        x = self.res_block3(x)
        x = self.gconv3(x, self.up8(self.conv1_l5(l5)))

        out = self.conv1_out(x)

        if grad is not None:
            out_grad = self.conv1_grad_out(torch.cat([out, grad], dim=1))
        else:
            out_grad = out

        return out, out_grad


class EMAModule(nn.Module):
    def __init__(self, channels, K, lbda=1, alpha=0.1, T=3):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.lbda = lbda
        self.conv1_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.bn_out = nn.BatchNorm2d(channels)
        self.register_buffer("bases", torch.empty(K, channels))  # K x C
        # self.bases = Parameter(torch.empty(K, channels), requires_grad=False) # K x C
        nn.init.kaiming_uniform_(self.bases, a=math.sqrt(5))
        # self.bases.data = F.normalize(self.bases.data, dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.conv1_in(x).view(B, C, -1).transpose(1, -1)  # B x N x C

        bases = self.bases[None, ...]
        x_in = x.detach()
        for i in range(self.T):
            # Expectation
            if i == (self.T - 1):
                x_in = x
            z = torch.softmax(
                self.lbda * torch.matmul(x_in, bases.transpose(1, -1)), dim=-1
            )  # B x N x K
            # Maximization
            bases = torch.matmul(z.transpose(1, 2), x_in) / (
                z.sum(1)[..., None] + 1e-12
            )  # B x K x C
            bases = F.normalize(bases, dim=-1)
        if self.training:
            self.bases.data = (1 - self.alpha) * self.bases + self.alpha * bases.detach().mean(0)

        x = torch.matmul(z, bases).transpose(1, -1).view(B, C, H, W)
        x = self.conv1_out(x)
        x = self.bn_out(x)
        x += residual
        return x


class ClassCenterBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, padding=0)

    def forward(self, features, segmap):
        B, _, H, W = features.shape
        K = segmap.shape[1]

        segmap = segmap.view(B, K, -1)

        features = self.conv1(features)
        features = features.view(B, self.channels, -1).transpose(1, 2)

        centers = torch.matmul(segmap, features)  # B, K, C
        centers = centers / (segmap.sum(-1, keepdim=True) + 1e-8)

        return centers


class ClassAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, centers, segmap):
        B, K, H, W = segmap.shape
        segmap = segmap.view(B, K, -1)

        centers = centers.transpose(1, 2)  # B, C, K

        out = torch.matmul(centers, segmap)  # B, C, HW
        out = out.view(B, self.channels, H, W)
        out = self.conv1(out)
        return out


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, b=1, gamma=2, k=None):
        super().__init__()
        if k is None:
            t = int(abs(math.log2(channels) + b / gamma))
            k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * y.expand_as(x)


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d"""

    def __init__(
        self,
        in_channels,
        channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        bias=True,
        radix=2,
        reduction_factor=4,
        norm_layer=None,
    ):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = nn.Conv2d(
            in_channels,
            channels * radix,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=bias,
        )
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(
            1.0 / self.p
        )


class HypercolumnsFusion(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        use_se_in=True,
        use_se_out=True,
        use_fusion_block=True,
        kz=3,
        relu=False,
        bn=False,
    ):
        super().__init__()
        self._use_se_in = use_se_in
        self._use_se_out = use_se_out
        self._use_fusion_block = use_fusion_block

        self.convs1 = nn.ModuleList(
            [
                ConvBn2d(in_c, channels, kernel_size=1, padding=1, relu=relu, bn=bn)
                for in_c in in_channels
            ]
        )

        if use_se_in:
            self.se_in = ChannelGate2d(channels, reduction=8)

        if use_se_out:
            self.se_out = ChannelGate2d(channels, reduction=8)

        if use_fusion_block:
            self.convs3 = nn.Sequential(
                ConvBn2d(
                    channels, channels, kernel_size=kz, padding=(kz - 1) // 2, relu=True, bn=True
                ),
                ConvBn2d(
                    channels, channels, kernel_size=kz, padding=(kz - 1) // 2, relu=True, bn=True
                ),
                ConvBn2d(
                    channels, channels, kernel_size=kz, padding=(kz - 1) // 2, relu=True, bn=True
                ),
            )

    def forward(self, feature_maps):
        size = feature_maps[-1].shape[2:]
        x = [
            F.upsample_bilinear(conv(fmap), size=size)
            for conv, fmap in zip(self.convs1, feature_maps)
        ]
        x = torch.stack(x, 0).sum(0)
        if self._use_se_in:
            x = self.se_in(x)
        if self._use_fusion_block:
            x = self.convs3(x)
        if self._use_se_out:
            x = self.se_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBn2d(
            in_channels, in_channels // 2, kernel_size=1, padding=0, relu=True, bn=True
        )
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1, padding=0)
        self.softplus = nn.Softplus()

    def forward(self, x):
        att = self.conv1(x)
        att = self.conv2(att)
        att = self.softplus(att)
        return att
