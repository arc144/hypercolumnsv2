from Core import losses

from ..BaseConfig import AttributeDict
from ..single_stage.UNet_R50 import UNet_R50_Config


class UNet_R50_Focal_Config(UNet_R50_Config):
    weights = {"ptx": None, "steel": None}

    loss = AttributeDict(loss_fun=losses.FocalLoss(alpha=0.25, gamma=4))
