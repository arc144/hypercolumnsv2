import numpy as np


def cmp_pos_dice(pred, gt, num_classes, iou_instead=False):
    pred = pred.reshape(pred.shape[0], -1)
    gt = gt.reshape(gt.shape[0], -1)

    pos_ix = gt.sum(1) > 0
    pred = pred[pos_ix]
    gt = gt[pos_ix]

    intersection = np.zeros((pred.shape[0], num_classes - 1), dtype=np.float64)
    union = np.zeros((pred.shape[0], num_classes - 1), dtype=np.float64)
    # We compute dice only for the classes of interest
    for c in range(1, num_classes):
        pred_c = pred == c
        gt_c = gt == c

        intersection[:, c - 1] = (pred_c * gt_c).astype(np.float64).sum(1)
        union[:, c - 1] = pred_c.astype(np.float64).sum(1) + gt_c.astype(np.float64).sum(1)

    if iou_instead:
        dice = intersection.sum(0) / (union.sum(0) - intersection.sum(0))
    else:
        dice = 2 * intersection.sum(0) / union.sum(0)

    return np.nanmean(dice)


def cmp_dataset_dice(pred, gt, num_classes, iou_instead=False):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    intersection = np.zeros((num_classes - 1), dtype=np.float64)
    union = np.zeros((num_classes - 1), dtype=np.float64)
    # We compute dice only for the classes of interest
    for c in range(1, num_classes):
        pred_c = pred == c
        gt_c = gt == c

        intersection[c - 1] = (pred_c * gt_c).astype(np.float64).sum()
        union[c - 1] = pred_c.astype(np.float64).sum() + gt_c.astype(np.float64).sum()

    if iou_instead:
        dice = intersection.sum(0) / (union.sum(0) - intersection.sum(0))
    else:
        dice = 2 * intersection.sum(0) / union.sum(0)

    # Now mean over classes
    dice = np.nanmean(dice)
    return dice


def cmp_balanced_cls_acc(pred, gt, num_classes, mean=False):
    gt_cls = np.zeros((pred.shape[0], num_classes))
    cls_pred = np.zeros((pred.shape[0], num_classes))
    for c in range(1, num_classes):
        gt_cls[:, c] = (gt == c).sum((1, 2)) > 0
        cls_pred[:, c] = (pred == c).sum((1, 2)) > 0

    # We consider as background images only when all other classes are void
    cls_pred[:, 0] = np.prod(np.logical_not(cls_pred[:, 1:]).astype(np.uint8), axis=1).astype(np.bool)
    gt_cls[:, 0] = np.prod(np.logical_not(gt_cls[:, 1:]).astype(np.uint8), axis=1).astype(np.bool)

    acc = np.logical_and(cls_pred, gt_cls).astype(np.uint8).sum(0) / gt_cls.astype(np.uint8).sum(0)
    if mean:
        return np.nanmean(acc)
    else:
        return acc


def cmp_binary_acc(pred, gt, threshold=0.5, mean=False):
    batch_size = pred.shape[0]
    gt = gt.reshape(batch_size, -1).sum(1) > 0
    pred = pred.squeeze() > threshold
    acc = pred == gt
    if mean:
        return acc.mean()
    else:
        return acc


def cmp_binary_acc_seg(pred, gt, mean=False):
    batch_size = pred.shape[0]
    gt = gt.reshape(batch_size, -1).sum(1) > 0
    pred = pred.reshape(batch_size, -1).sum(1) > 0
    acc = pred == gt
    if mean:
        return acc.mean()
    else:
        return acc


def cmp_balanced_binary_acc_seg(pred, gt):
    batch_size = pred.shape[0]
    gt = gt.reshape(batch_size, -1).sum(1) > 0
    pred = pred.reshape(batch_size, -1).sum(1) > 0
    acc1 = pred[gt == True] == True
    acc0 = pred[gt == False] == False
    return (acc0.mean() + acc1.mean()) / 2
