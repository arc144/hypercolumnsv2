import Utils.augmentations as Aug
import Utils.TTA as TTA
from Core import losses
from Models.models import *

from ..BaseConfig import AConfig, AttributeDict, TrainingCycle


class UNet_R50_SegOnly_Config(AConfig):
    seed = 2019
    task = "seg"
    weights = {
        "ptx": None,
        "steel": None,
        "motocar": None,
    }

    # Model parameters
    model = AttributeDict(
        architecture=UNet_R50,
        pretrained=True,
    )

    dataloader = AttributeDict(
        workers=4,
        balanced_sampler=False,
    )

    loss = AttributeDict(loss_fun=losses.CrossEntropy())

    optim = AttributeDict(
        type="RAdam",
        momentum=0.9,
        weight_decay=0.0001,
    )

    train_cycles = {
        "ptx": [
            TrainingCycle(
                lr=1e-3,
                n_epoch=30,
                tgt_size=(512, 512),
                lr_min=1e-6,
                scheduler="CosineAnnealing",
                tmax=30,
                tmul=1,
                mixup=0.0,
                aug=Aug.Aug0,
                batch_size=16,
                ema=False,
                remove_bg_only=True,
                start_val_on_epoch=0,
            )
        ],
        "motocar": [
            TrainingCycle(
                lr=2e-3,
                n_epoch=37,
                tgt_size=(384, 384),
                lr_min=1e-6,
                scheduler="CosineAnnealing",
                tmax=37,
                tmul=1,
                mixup=0.0,
                aug=Aug.Aug0,
                batch_size=32,
                ema=False,
                remove_bg_only=True,
                start_val_on_epoch=0,
            )
        ],
    }

    fold = 11
    tta = [TTA.Nothing()]
