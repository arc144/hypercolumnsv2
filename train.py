import imgaug as ia
import numpy as np
import torch
from fire import Fire

# from apex import amp
from Core.dataset import DatasetFactory
from Core.tasks import BaseNetwork
from Experiments import (
    EffDetD2_Config,
    FaPN_R50_Config,
    ISUEI_D2_Config,
    ISUEI_FaPN_R50_Config,
    ISUEI_UNet_R50_Config,
    ISUEI_UNet_R50_Focal_Config,
    R50_Cls_Config,
    UNet_HC_R50_Config,
    UNet_R50_Config,
    UNet_R50_Focal_Config,
    UNet_R50_Oversampling_Config,
    UNet_R50_SegOnly_Config,
    UNet_R50_Undersampling_Config,
    UNetR50_RandomSampling_Config,
)

# fmt: off
available_experiments = {
    # two stages experiments
    "r50_classification": R50_Cls_Config,
    "unet_r50_segmentation": UNet_R50_SegOnly_Config, # this is trained for seg only on foreground imgs

    # single stage experiments
    # unet variants
    "unet_r50": UNet_R50_Config,
    "unet_hc_r50": UNet_HC_R50_Config,
    "isuei_unet_r50": ISUEI_UNet_R50_Config,
    # efficient det variants
    "effdet_d2": EffDetD2_Config,
    "isuei_d2": ISUEI_D2_Config,
    # fapn variants
    "fapn_r50": FaPN_R50_Config,
    "isuei_fapn_r50": ISUEI_FaPN_R50_Config,

    # Focal loss experiments
    "isuei_unet_r50_focal": ISUEI_UNet_R50_Focal_Config,
    "unet_r50_focal": UNet_R50_Focal_Config,

    # sampling experiments
    "unet_r50_oversampling": UNet_R50_Oversampling_Config,
    "unet_r50_randomsampling": UNetR50_RandomSampling_Config,
    "unet_r50_undersampling": UNet_R50_Undersampling_Config,
}
# fmt: on


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ia.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(exp_name: str, dataset_name: str):
    cfg = available_experiments[exp_name]

    set_seeds(cfg.seed + cfg.fold)
    dataset_cfg = getattr(cfg.datasets, dataset_name)
    dataset = DatasetFactory(dataset_cfg.train.csv_path, dataset_cfg.train.root, dataset_name)

    net = cfg.model.architecture(
        num_classes=dataset_cfg.num_classes,
        pretrained=cfg.model.pretrained,
        alpha=cfg.loss.setdefault("alpha", None),
        beta=cfg.loss.setdefault("beta", None),
    )

    trainer = BaseNetwork(
        net, task=cfg.task, mode="train", criterion=cfg.loss.loss_fun, debug=False, fold=cfg.fold
    )

    if cfg.weights.get(dataset_name, None) is not None:
        trainer.load_model(cfg.weights[dataset_name])

    optims, schedulers = [], []
    for cycle in cfg.train_cycles[dataset_name]:
        optimizer, scheduler = trainer.create_optmizer(
            optimizer=cfg.optim.type,
            lr=cycle.lr,
            scheduler=cycle.scheduler,
            T_max=cycle.tmax,
            T_mul=cycle.tmul,
            lr_min=cycle.lr_min,
            freeze_encoder=cycle.freeze_encoder,
            freeze_bn=cycle.freeze_bn,
        )
        optims.append(optimizer)
        schedulers.append(scheduler)

    # trainer.net, optims = amp.initialize(
    #     trainer.net, optims, opt_level=cfg.apex if hasattr(cfg, "apex") else "O0"
    # )

    for cycle, optim, sch in zip(cfg.train_cycles[dataset_name], optims, schedulers):
        val_loader = dataset.yield_loader(
            is_test=True,
            remove_bg_only=cycle.remove_bg_only,
            tgt_size=cycle.tgt_size,
            batch_size=cycle.batch_size,
            workers=cfg.dataloader.workers,
        )

        train_loader = dataset.yield_loader(
            is_test=False,
            tgt_size=cycle.tgt_size,
            batch_size=cycle.batch_size,
            aug=cycle.aug,
            remove_bg_only=cycle.remove_bg_only,
            balanced_sampler=cfg.dataloader.balanced_sampler,
            pos_ratio_range=cycle.pos_ratio_range,
            sampling_mode=cycle.sampling_mode,
            epochs=cycle.n_epoch,
            workers=cfg.dataloader.workers,
        )
        trainer.train_network(
            train_loader,
            val_loader,
            optim,
            sch,
            grad_acc=cycle.grad_acc,
            n_epoch=cycle.n_epoch,
            epoch_per_val=cycle.epoch_per_val,
            start_val_on_epoch=cycle.start_val_on_epoch,
            ema=cycle.ema,
            ema_start=cycle.ema_start,
            ema_decay=cycle.ema_decay,
            mixup=cycle.mixup,
        )


if __name__ == "__main__":
    Fire(main)
