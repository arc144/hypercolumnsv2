import os

import torch
import Utils.augmentations as Aug
import Utils.TTA as TTA
from Core import losses
from Models.models import *


class AttributeDict(dict):
    __getattr__ = dict.__getitem__


class TrainingCycle:
    def __init__(
        self,
        lr,
        n_epoch,
        tgt_size=(256, 256),
        aug=Aug.Aug0,
        mixup=0.2,
        batch_size=32,
        lr_min=0.0,
        scheduler=None,
        tmax=31,
        tmul=1,
        grad_acc=1,
        pos_ratio_range=(0.2, 0.8),
        sampling_mode="mixed",
        ema=False,
        ema_start=0,
        ema_decay=0.99,
        remove_bg_only=False,
        freeze_encoder=False,
        freeze_bn=False,
        epoch_per_val=1,
        start_val_on_epoch=0,
        shuffle=True,
    ):
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema = ema
        self.pos_ratio_range = pos_ratio_range
        self.sampling_mode = sampling_mode
        self.remove_bg_only = remove_bg_only
        self.batch_size = batch_size
        self.mixup = mixup
        self.aug = aug
        self.tgt_size = tgt_size
        self.start_val_on_epoch = start_val_on_epoch
        self.shuffle = shuffle
        self.epoch_per_val = epoch_per_val
        self.freeze_bn = freeze_bn
        self.freeze_bn = freeze_bn
        self.freeze_encoder = freeze_encoder
        self.grad_acc = grad_acc
        self.tmul = tmul
        self.tmax = tmax
        self.scheduler = scheduler
        self.n_epoch = n_epoch
        self.lr_min = lr_min
        self.lr = lr


class AConfig(object):
    seed = 2019
    task = "seg"
    weights = None

    dataloader = AttributeDict(
        workers=5,
        balanced_sampler=True,
    )

    loss = losses.CrossEntropy()

    optim = AttributeDict(
        type="RAdam",
        momentum=0.9,
        weight_decay=0.0001,
    )

    train_cycles = [
        TrainingCycle(
            lr=5e-4,
            n_epoch=75,
            tgt_size=(128, 800),
            lr_min=1e-5,
            scheduler="CosineAnnealing",
            tmax=5,
            tmul=2,
            mixup=0.2,
            batch_size=64,
            aug=Aug.Aug0,
            pos_ratio_range=(0.6, 0.6),
            start_val_on_epoch=0,
        ),
    ]

    fold = 0

    datasets = AttributeDict(
        ptx=AttributeDict(
            num_classes=2,
            train=AttributeDict(
                csv_path="./Data/ptx/Folds/fold0.csv",
                root="/media/nvme/Datasets/Pneumothorax/dicom-images-train",
                val_mode="max",
            ),
            test=AttributeDict(
                csv_path="./Data/ptx/Folds/holdout.csv",
                root="/media/nvme/Datasets/Pneumothorax/dicom-images-train",
                out_path="./Output/ptx/",
            ),
        ),
        motocar=AttributeDict(
            num_classes=3,
            train=AttributeDict(
                csv_path="/media/nvme/Datasets/COCO/annotations/instances_train2017.json",
                root="/media/nvme/Datasets/COCO/train2017",
                val_mode="max",
            ),
            test=AttributeDict(
                csv_path="/media/nvme/Datasets/COCO/annotations/instances_val2017.json",
                root="/media/nvme/Datasets/COCO/val2017",
                out_path="./Output/ptx/",
            ),
        ),
    )

    tta = [TTA.Nothing()]

    def __init__(self):
        self.seed = self.seed + self.fold
        print("Fold {} selected".format(self.fold))

        for dset in ["steel", "ptx"]:
            train_path = getattr(self.datasets, dset).train.csv_path
            test_path = getattr(self.datasets, dset).test.csv_path
            out_path = getattr(self.datasets, dset).test.out_path

            # Check if paths exist
            for p in [train_path, test_path]:
                assert os.path.exists(p), "Path does not exists. Got {}".format(p)
            # Check if out_path exist and create it
            if not os.path.exists(out_path):
                os.mkdir(out_path)
