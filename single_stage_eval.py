from fire import Fire

from Core import metrics
from Core.dataset import DatasetFactory
from Core.tasks import BaseNetwork
from train import available_experiments


def validation(pred, mask, num_classes):
    dice_pos = metrics.cmp_pos_dice(pred, mask, num_classes=num_classes, iou_instead=True)
    dice = metrics.cmp_dataset_dice(pred, mask, num_classes=num_classes, iou_instead=True)
    cls_acc = metrics.cmp_balanced_cls_acc(pred, mask, num_classes=num_classes, mean=True)
    bin_acc = metrics.cmp_balanced_binary_acc_seg(pred, mask)
    return dice, dice_pos, cls_acc, bin_acc


def main(exp_name: str, dataset_name: str, model_weights_path: str):
    cfg = available_experiments[exp_name]

    dataset_cfg = getattr(cfg.datasets, dataset_name)
    dataset = DatasetFactory(dataset_cfg.test.csv_path, dataset_cfg.test.root, dataset_name)

    test_loader = dataset.yield_loader(
        is_test=True, tgt_size=cfg.train_cycles[dataset_name][-1].tgt_size, batch_size=8, workers=6
    )

    model = cfg.model.architecture(
        num_classes=dataset_cfg.num_classes,
        pretrained=cfg.model.pretrained,
        alpha=cfg.loss.setdefault("alpha", None),
        beta=cfg.loss.setdefault("beta", None),
    )

    net = BaseNetwork(model, mode="test", criterion=cfg.loss.loss_fun, debug=False, fold=cfg.fold)

    assert model_weights_path is not None, "Model weights is None"
    net.load_model(model_weights_path)

    image_ids, pred, mask = net.predict(test_loader, pbar=True)

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
