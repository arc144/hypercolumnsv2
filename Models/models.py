import torch
import torch.nn as nn
import torch.nn.functional as F
from Backbone import *
from Core.losses import dice_score

import Models.layers as L

from .efficientdet.effdet.factory import create_model

try:
    from .fapn import FAN
except ModuleNotFoundError:
    import warnings

    warnings.warn("Could not import FAPN model. Please ensure you have it installed")

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class R34(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, **kwargs):
        super(R34, self).__init__()
        self.num_classes = num_classes
        self.encoder = resnet34(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1, e2, e3, e4 = self.encoder(x)
        p = self.pool(e4)
        logit = self.fc(p.squeeze(3).squeeze(2))
        return [logit]

    def loss(self, criterion, logit, mask):
        cls_mask = (mask.view(mask.shape[0], -1).sum(1) > 0).float()
        cls_loss = criterion(logit[0], cls_mask.unsqueeze(1))
        return cls_loss.mean()


class R50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, **kwargs):
        super(R50, self).__init__()
        self.num_classes = num_classes
        self.encoder = resnet50(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3, inplace=True)
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1, e2, e3, e4 = self.encoder(x)
        p = self.pool(e4)
        p = self.dropout(p)
        logit = self.fc(p.squeeze(3).squeeze(2))
        return [logit]

    def loss(self, criterion, logit, mask):
        cls_mask = (mask.view(mask.shape[0], -1).sum(1) > 0).float()
        cls_loss = criterion(logit[0], cls_mask.unsqueeze(1))
        return cls_loss.mean()


class UNet_R50(nn.Module):
    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = resnet50(pretrained=pretrained)

        self.decoder1 = L.UNetDecoder(2048 + 1024, 1024, 512)
        self.decoder2 = L.UNetDecoder(512 + 512, 512, 256)
        self.decoder3 = L.UNetDecoder(256 + 256, 256, 128)

        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.logit = nn.Conv2d(128, self.num_classes, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1, e2, e3, e4 = self.encoder(x)

        d = self.decoder1(e4, skip=e3)
        d = self.decoder2(d, skip=e2)
        d = self.decoder3(d, skip=e1)

        logit = self.upsample(self.logit(d))

        return [logit]

    def loss(self, criterion, logit, mask):
        loss = criterion(logit[0], mask)
        return loss


class UNetHC_R50(nn.Module):
    def __init__(self, num_classes, pretrained=True, alpha=0.05, beta=0.01, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.encoder = resnet50(pretrained=pretrained)

        self.decoder1 = L.UNetDecoder(2048 + 1024, 1024, 512)
        self.decoder2 = L.UNetDecoder(512 + 512, 512, 256)
        self.decoder3 = L.UNetDecoder(256 + 256, 256, 128)

        self.hc = L.HypercolumnsFusion(
            [2048, 512, 256, 128],
            128,
            kz=3,
            use_fusion_block=False,
            use_se_in=False,
            use_se_out=False,
        )

        self.seg_logit = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1, e2, e3, e4 = self.encoder(x)

        d3 = self.decoder1(e4, skip=e3)
        d2 = self.decoder2(d3, skip=e2)
        d1 = self.decoder3(d2, skip=e1)

        base_feat = self.hc([e4, d3, d2, d1])
        seg_logit = self.seg_logit(base_feat)

        return [self.upsample(seg_logit)]

    def loss(self, criterion, logit, mask):
        loss = criterion(logit[0], mask)
        return loss


class ISUEI_UNet_R50(nn.Module):
    def __init__(self, num_classes, pretrained=True, alpha=0.05, beta=0.01, use_hc=True, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.use_hc = use_hc
        self.encoder = resnet50(pretrained=pretrained)

        self.decoder1 = L.UNetDecoder(2048 + 1024, 1024, 512)
        self.decoder2 = L.UNetDecoder(512 + 512, 512, 256)
        self.decoder3 = L.UNetDecoder(256 + 256, 256, 128)

        if self.use_hc:
            self.hc = L.HypercolumnsFusion(
                [2048, 512, 256, 128],
                128,
                kz=3,
                use_fusion_block=True,
                use_se_in=True,
                use_se_out=True,
            )

        self.cls_logit = nn.Conv2d(128, self.num_classes - 1, kernel_size=1)
        self.seg_logit = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1, e2, e3, e4 = self.encoder(x)

        d3 = self.decoder1(e4, skip=e3)
        d2 = self.decoder2(d3, skip=e2)
        d1 = self.decoder3(d2, skip=e1)

        if self.use_hc:
            base_feat = self.hc([e4, d3, d2, d1])
        else:
            base_feat = d1

        cls_logit = self.cls_logit(self.pool(base_feat))
        seg_logit = self.seg_logit(base_feat)

        return [self.upsample(seg_logit), cls_logit]

    def loss(self, criterion, logit, mask):
        loss = criterion(logit[0], mask)
        cls_mask = torch.zeros((mask.shape[0], self.num_classes, *mask.shape[1:]))
        cls_mask.scatter_(1, mask.unsqueeze(1), 1)
        cls_mask = cls_mask[:, 1:, :, :]
        cls_mask = (cls_mask.view(*cls_mask.shape[:2], -1).sum(2) > 0).float()
        cls_loss = F.binary_cross_entropy_with_logits(
            logit[1].squeeze(), cls_mask.squeeze(), reduction="none"
        )
        cls_loss = self.beta * cls_loss.mean()

        pos_ix = mask.view(cls_mask.shape[0], -1).sum(1) > 0
        if pos_ix.sum() > 0:
            seg_loss = 1 - dice_score(logit[0][pos_ix], mask[pos_ix], per_image=True, smooth=1)
            seg_loss = self.alpha * seg_loss
        else:
            seg_loss = 0

        return loss + cls_loss + seg_loss
        # return loss


class EfficientDetD2(nn.Module):
    def __init__(self, num_classes, pretrained=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        model = create_model("tf_efficientdet_d2", pretrained=pretrained)
        self.encoder = model.backbone
        self.decoder = model.fpn

        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.logit = nn.Conv2d(112, self.num_classes, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.encoder(x)
        x = self.decoder(x)
        seg_logit = self.logit(x[0])
        return [self.upsample(seg_logit)]

    def loss(self, criterion, logit, mask):
        loss = criterion(logit[0], mask)
        return loss


class ISUEI_D2(nn.Module):
    def __init__(self, num_classes, pretrained=True, alpha=0.05, beta=0.01, use_hc=True, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.use_hc = use_hc

        model = create_model("tf_efficientdet_d2", pretrained=pretrained)
        self.encoder = model.backbone
        self.decoder = model.fpn

        if self.use_hc:
            self.hc = L.HypercolumnsFusion(
                [112, 112, 112, 112, 112],
                112,
                kz=3,
                use_fusion_block=True,
                use_se_in=True,
                use_se_out=True,
            )

        self.cls_logit = nn.Conv2d(112, self.num_classes - 1, kernel_size=1)
        self.seg_logit = nn.Conv2d(112, self.num_classes, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.encoder(x)
        p3, p4, p5, p6, p7 = self.decoder(x)

        if self.use_hc:
            base_feat = self.hc([p7, p6, p5, p4, p3])
        else:
            base_feat = p3

        cls_logit = self.cls_logit(self.pool(base_feat))
        seg_logit = self.seg_logit(base_feat)

        return [self.upsample(seg_logit), cls_logit]

    def loss(self, criterion, logit, mask):
        loss = criterion(logit[0], mask)
        cls_mask = torch.zeros((mask.shape[0], self.num_classes, *mask.shape[1:]))
        cls_mask.scatter_(1, mask.unsqueeze(1), 1)
        cls_mask = cls_mask[:, 1:, :, :]
        cls_mask = (cls_mask.view(*cls_mask.shape[:2], -1).sum(2) > 0).float()
        cls_loss = F.binary_cross_entropy_with_logits(
            logit[1].squeeze(), cls_mask.squeeze(), reduction="none"
        )
        cls_loss = self.beta * cls_loss.mean()

        pos_ix = mask.view(cls_mask.shape[0], -1).sum(1) > 0
        if pos_ix.sum() > 0:
            seg_loss = 1 - dice_score(logit[0][pos_ix], mask[pos_ix], per_image=True, smooth=1)
            seg_loss = self.alpha * seg_loss
        else:
            seg_loss = 0

        return loss + cls_loss + seg_loss


class FaPN_R50(nn.Module):
    def __init__(self, num_classes, pretrained=True, alpha=0.05, beta=0.01, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.encoder = resnet50(pretrained=pretrained)

        strides = [4, 8, 16, 32]
        in_channels_per_feat = [256, 512, 1024, 2048]

        self.decoder = FAN(
            strides=strides, in_channels_per_feature=in_channels_per_feat, out_channels=128
        )

        self.seg_logit = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1, e2, e3, e4 = self.encoder(x)

        d = self.decoder([e1, e2, e3, e4])

        seg_logit = self.seg_logit(d["p2"])

        return [self.upsample(seg_logit)]

    def loss(self, criterion, logit, mask):
        loss = criterion(logit[0], mask)
        return loss


class ISUEI_FaPN_R50(nn.Module):
    def __init__(self, num_classes, pretrained=True, alpha=0.05, beta=0.01, use_hc=True, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.use_hc = use_hc

        self.encoder = resnet50(pretrained=pretrained)

        strides = [4, 8, 16, 32]
        in_channels_per_feat = [256, 512, 1024, 2048]

        self.decoder = FAN(
            strides=strides, in_channels_per_feature=in_channels_per_feat, out_channels=128
        )

        if self.use_hc:
            self.hc = L.HypercolumnsFusion(
                [128, 128, 128], 128, kz=3, use_fusion_block=True, use_se_in=True, use_se_out=True
            )

        self.cls_logit = nn.Conv2d(128, self.num_classes - 1, kernel_size=1)
        self.seg_logit = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d(1)

        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1, e2, e3, e4 = self.encoder(x)
        d = self.decoder([e1, e2, e3, e4])

        if self.use_hc:
            base_feat = self.hc([d["p4"], d["p3"], d["p2"]])
        else:
            base_feat = d["p2"]

        cls_logit = self.cls_logit(self.pool(base_feat))
        seg_logit = self.seg_logit(base_feat)

        return [self.upsample(seg_logit), cls_logit]

    def loss(self, criterion, logit, mask):
        loss = criterion(logit[0], mask)
        cls_mask = torch.zeros((mask.shape[0], self.num_classes, *mask.shape[1:]))
        cls_mask.scatter_(1, mask.unsqueeze(1), 1)
        cls_mask = cls_mask[:, 1:, :, :]
        cls_mask = (cls_mask.view(*cls_mask.shape[:2], -1).sum(2) > 0).float()
        cls_loss = F.binary_cross_entropy_with_logits(
            logit[1].squeeze(), cls_mask.squeeze(), reduction="none"
        )
        cls_loss = self.beta * cls_loss.mean()

        pos_ix = mask.view(cls_mask.shape[0], -1).sum(1) > 0
        if pos_ix.sum() > 0:
            seg_loss = 1 - dice_score(logit[0][pos_ix], mask[pos_ix], per_image=True, smooth=1)
            seg_loss = self.alpha * seg_loss
        else:
            seg_loss = 0

        return loss + cls_loss + seg_loss
