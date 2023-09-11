import Utils.augmentations as Aug
from Core import losses

from ..BaseConfig import AttributeDict
from ..single_stage.ISUEI_UNet_R50 import ISUEI_UNet_R50_Config


class ISUEI_UNet_R50_Focal_Config(ISUEI_UNet_R50_Config):

    weights = {
        "ptx": None,
        "motocar": None,
    }

    loss = AttributeDict(
        loss_fun=losses.FocalLoss(alpha=0, gamma=4),
        alpha=0.0,
        beta=0.0,
    )

    apex = "O2"
