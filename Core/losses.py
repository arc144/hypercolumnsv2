import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode="bilinear")

        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000 / 4, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode="bilinear")
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = (
            pred.contiguous()
            .view(
                -1,
            )[mask]
            .contiguous()
            .sort()
        )
        min_value = pred[int(min(self.min_kept, pred.numel() - 1))]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()


class WeightedBCE(nn.Module):
    def __init__(self, pos_w=0.25, neg_w=0.75):
        super().__init__()
        self.neg_w = neg_w
        self.pos_w = pos_w

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert logit.shape == truth.shape

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction="none")

        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25 * pos * loss / pos_weight + 0.75 * neg * loss / neg_weight).sum()

        return loss


################### DICE ########################
def dice_score(logit, truth, per_image=False, smooth=2):
    prob = torch.softmax(logit, dim=1)

    oh_truth = torch.zeros_like(prob)
    oh_truth.scatter_(1, truth.unsqueeze(1), 1)

    # Remove bg from loss
    prob = prob[:, 1:, :, :]
    oh_truth = oh_truth[:, 1:, :, :]

    # Compute dice per batch
    B = 1
    if per_image:
        B = oh_truth.shape[0]

    prob = prob.contiguous().view(B, -1)
    oh_truth = oh_truth.contiguous().view(B, -1)

    intersection = (prob * oh_truth).sum(1)
    union = prob.sum(1) + oh_truth.sum(1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, per_image=False):
        super(DiceLoss, self).__init__()
        self.per_image = per_image
        self.smooth = smooth

    def forward(self, logit, truth):
        dice = dice_score(logit, truth, self.per_image, self.smooth)
        loss = 1 - dice
        return loss


################ FOCAL LOSS ####################
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


################# BCE + DICE ########################
class CE_Dice(nn.Module):
    def __init__(self, smooth=1, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.dice = DiceLoss(smooth=smooth)
        self.bce = CrossEntropy()

    def forward(self, logit, truth):
        dice = self.dice(logit, truth)
        bce = self.bce(logit, truth)
        if np.random.rand() < 0.001:
            print(
                "bce: {:.3f} alpha_dice {:.3f} dice: {:.3f}".format(bce, self.alpha * dice, dice)
            )
        return self.alpha * dice + bce
