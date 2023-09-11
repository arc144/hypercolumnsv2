import gc
import os

import numpy as np
from fire import Fire

from Core import metrics
from Core.dataset import DatasetFactory
from Core.tasks import BaseNetwork
from single_stage_eval import validation
from train import available_experiments


def create_model_and_predict(cfg, weights_path, loader, num_classes, dataset_name):
    model = cfg.model.architecture(
        num_classes=num_classes,
        pretrained=cfg.model.pretrained,
        alpha=cfg.loss.setdefault("alpha", None),
        beta=cfg.loss.setdefault("beta", None),
    )

    net = BaseNetwork(
        model, task=cfg.task, mode="test", criterion=cfg.loss.loss_fun, debug=False, fold=cfg.fold
    )

    assert weights_path is not None, "Model weights is None"
    net.load_model(weights_path)

    image_ids, pred, mask = net.predict(loader, pbar=True)
    return image_ids, pred, mask


def main(
    dataset_name: str,
    cls_exp_name: str,
    cls_weights_path: str,
    seg_exp_name: str,
    seg_weights_path: str,
):
    cls_cfg = available_experiments[cls_exp_name]
    seg_cfg = available_experiments[seg_exp_name]

    dataset_cfg = getattr(seg_cfg.datasets, dataset_name)
    dataset = DatasetFactory(dataset_cfg.test.csv_path, dataset_cfg.test.root, dataset_name)

    test_loader = dataset.yield_loader(
        is_test=True,
        tgt_size=seg_cfg.train_cycles[dataset_name][-1].tgt_size,
        batch_size=8,
        workers=6,
    )

    image_ids, pred, mask = create_model_and_predict(
        seg_cfg, seg_weights_path, test_loader, dataset_cfg.num_classes, dataset_name
    )
    test_loader = dataset.yield_loader(
        is_test=True,
        tgt_size=cls_cfg.train_cycles[dataset_name][-1].tgt_size,
        batch_size=8,
        workers=6,
    )

    _, cls, _ = create_model_and_predict(
        cls_cfg, cls_weights_path, test_loader, dataset_cfg.num_classes, dataset_name
    )

    # Set to empty predictions where classification said it was empty
    pred[~cls.astype(bool)] = 0

    dice, dice_pos, cls_acc, bin_acc = validation(pred, mask, dataset_cfg.num_classes)
    print(
        "IoU:{:.3f}\tIoU_pos:{:.3f}\tClass accuracy:{:.3f}\tBinary accuracy:{:.3f}".format(
            dice,
            dice_pos,
            cls_acc,
            bin_acc,
        )
    )


if __name__ == "__main__":
    Fire(main)
