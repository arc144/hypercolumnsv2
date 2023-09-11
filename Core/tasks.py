import copy
import datetime
import gc
import math
import os
import time
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorboardX as tbX
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler

# from apex.amp import scale_loss
from tqdm import tqdm
from Utils import TTA
from Utils import data_helpers as DH
from Utils import net_helpers as H

from Core import metrics
from Core import scheduler as sch
from Core.optmizers import AdamW, RAdam

scaler = GradScaler()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class BaseNetwork:
    """Base model for Networks.
    Subclass it and override methods for task specific metrics
    and funcionalities"""

    def __init__(
        self,
        net,
        task="seg",
        mode="train",
        criterion=None,
        fold=None,
        debug=False,
        val_mode="max",
        comment="",
    ):
        super().__init__()
        self.task = task
        self.fold = fold
        self.debug = debug
        self.best_model_path = None
        self.epoch = 0
        self.val_mode = val_mode
        self.comment = comment
        self.is_training = False
        self.criterion = criterion
        self.tta = [TTA.Nothing()]
        self.net = net
        self.freeze_encoder = False
        self.freeze_bn = False
        self.ema_model = None

        if self.val_mode == "max":
            self.best_metric = -np.inf
        elif self.val_mode == "min":
            self.best_metric = np.inf

        self.train_log = {}
        self.val_log = {}
        if mode == "train":
            self.create_save_folder()
            self.writer = tbX.SummaryWriter(log_dir=self.save_dir)

    # TODO: Implment a lr_finder method.

    def create_optmizer(
        self,
        optimizer="SGD",
        lr=1e-3,
        scheduler=None,
        gamma=0.25,
        patience=4,
        milestones=None,
        T_max=10,
        T_mul=2,
        lr_min=0,
        freeze_encoder=False,
        freeze_bn=False,
        weight_decay=0,
    ):
        self.freeze_bn = freeze_bn
        self.net.cuda()
        # self.set_encoder_trainable(not freeze_encoder)
        parameters = []
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            # elif name.startswith('encoder'):
            #     parameters.append({"params": param, "lr": 0.1 * self.lr})
            # param.requires_grad = False
            else:
                if name.endswith("weight"):
                    parameters.append({"params": param})
                else:
                    parameters.append({"params": param, "weight_decay": 0})

        if optimizer == "SGD":
            optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

        elif optimizer == "Adam":
            optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)

        elif optimizer == "RAdam":
            optimizer = RAdam(parameters, lr=lr, weight_decay=weight_decay)

        if scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=gamma,
                patience=patience,
                verbose=True,
                threshold=0.01,
                min_lr=1e-05,
                eps=1e-08,
            )

        elif scheduler == "Milestones":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma, last_epoch=-1
            )

        elif scheduler == "CosineAnnealing":
            scheduler = sch.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                T_mul=T_mul,
                lr_min=lr_min,
                val_mode=self.val_mode,
                last_epoch=-1,
                save_snapshots=True,
                save_all=True,
            )
        elif scheduler == "OneCycleLR":
            scheduler = sch.OneCycleLR(optimizer, num_steps=T_max, lr_range=(lr_min, lr))

        elif scheduler == "Exponential":
            exp_decay = math.exp(-0.01)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

        elif isinstance(scheduler, str):
            raise NotImplementedError("Scheduler not recognized.")

        return optimizer, scheduler

    ######################### TRAINING #############################
    def train_network(
        self,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        grad_acc=1,
        n_epoch=10,
        start_val_on_epoch=0,
        epoch_per_val=1,
        ema=False,
        ema_start=0,
        ema_decay=0.99,
        mixup=0,
    ):
        with H.timer("Training Fold {}".format(self.fold)):
            # Required to correct behavior when resuming training
            n_epoch = n_epoch + self.epoch
            print(
                "Model created, total of {} parameters".format(
                    sum(p.numel() for p in self.net.parameters())
                )
            )

            while self.epoch < n_epoch:
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(self.epoch)
                self.epoch += 1
                lr = np.mean([param_group["lr"] for param_group in optimizer.param_groups])
                # self.do_validation(self.net, val_loader)
                with H.timer("Train Epoch {:}/{:} - LR: {:.4E}".format(self.epoch, n_epoch, lr)):
                    # Training step
                    self.training_step(train_loader, optimizer, grad_acc=grad_acc, mixup=mixup)

                    # Update ema model
                    if ema:
                        self.update_ema(decay=ema_decay if self.epoch > ema_start else 0)

                    #  Validation
                    if self.epoch == 1 or (
                        not (self.epoch - 1) % epoch_per_val and self.epoch > start_val_on_epoch
                    ):
                        self.do_validation(self.ema_model if ema else self.net, val_loader)
                    # Learning Rate Scheduler
                    if scheduler is not None:
                        if type(scheduler).__name__ == "ReduceLROnPlateau":
                            scheduler.step(self.val_log["total_mAP"][-1])
                        elif type(scheduler).__name__ == "CosineAnnealingLR":
                            self.best_model_path = scheduler.step(
                                self.epoch,
                                save_dict=dict(
                                    dice=self.val_log["sum"][-1],
                                    save_dir=self.save_dir,
                                    fold=self.fold,
                                    state_dict=self.ema_model.state_dict()
                                    if ema
                                    else self.net.state_dict(),
                                ),
                            )
                        else:
                            scheduler.step()
                    # Save best model
                    if type(scheduler).__name__ != "CosineAnnealingLR":
                        self.save_best_model(
                            self.ema_model if ema else self.net, self.val_log["sum"][-1]
                        )

                self.update_tbX(self.epoch, optimizer)
            # self.save_training_log()
            self.writer.close()

    def training_step(self, train_loader, optimizer, grad_acc=1, ths=0.5, noise_th=0, mixup=0):
        """Training step of a single epoch"""
        self.set_mode("train")
        # Define the frequency metrics are computed
        n_iter = len(train_loader.batch_sampler.sampler) / train_loader.batch_sampler.batch_size

        # Begin epoch loop
        loss_list = []
        dice_list = []
        acc_list = []
        pbar = tqdm(enumerate(train_loader), total=n_iter)
        optimizer.zero_grad()
        for i, (im_id, im, mask) in pbar:
            # bg = torch.sum(torch.mean((mask == 0).float(), [1, 2]) == 1).numpy()
            # c1 = torch.sum(torch.sum(mask == 1, [1, 2]) > 0).numpy()
            # c2 = torch.sum(torch.sum(mask == 2, [1, 2]) > 0).numpy()
            # print(bg, c1, c2)
            # import matplotlib.pyplot as plt
            # palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
            # fig, axis = plt.subplots(16, 1, figsize=(100, 100))
            # axis = axis.ravel()
            # for i in range(16):
            #     im_i = np.moveaxis((im[i].numpy() * 255).astype(np.uint8), 0, -1)
            #     im_i2 = im_i.copy()
            #     mask_i = mask[i].numpy().squeeze()
            #     for c in range(1, 5):
            #         ix = np.where(mask_i == c)
            #         im_i2[ix] = palet[c - 1]
            #     axis[i].imshow(im_i)
            #     axis[i].imshow(im_i2, alpha=0.3)
            # plt.show()

            im = im.cuda()
            mask = mask.cuda()
            # Mixup?
            if mixup > 0:
                im, mask, mask_b, lam = DH.mixup_data(im, mask, alpha=mixup)
            # Forward propagation
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logit = self.net(im)
                loss = self.net.loss(self.criterion, logit, mask)
                if mixup > 0:
                    loss = lam * loss + (1 - lam) * self.net.loss(self.criterion, logit, mask_b)
                loss = loss.mean()
            loss_list.append(loss.item())

            scaler.scale(loss).backward()

            if self.task == "seg":
                pred = torch.argmax(logit[0], 1).data.cpu().numpy()
                acc = metrics.cmp_balanced_cls_acc(
                    pred, mask.data.cpu().numpy(), self.net.num_classes, mean=True
                )
                dice = metrics.cmp_pos_dice(
                    pred, mask.data.cpu().numpy(), self.net.num_classes, iou_instead=True
                ).mean()
            else:
                pred = torch.sigmoid(logit[0]).data.cpu().numpy()
                acc = metrics.cmp_binary_acc(pred, mask.data.cpu().numpy()).mean()
                dice = acc

            acc_list.append(acc)
            dice_list.append(dice)
            pbar.set_postfix_str(
                "loss: {:.3f} dice: {:.3f}, acc: {:.3f}".format(loss.item(), dice, acc)
            )

            if (i + 1) % grad_acc == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()

        # Append epoch data to metrics dict
        out = dict(loss=loss_list, dice=dice_list)

        for metric, value in out.items():
            H.update_log(self.train_log, metric, np.mean(value))

    def do_validation(
        self,
        model,
        val_loader,
        ths=0.5,
        size_th=(600, 600, 1000, 2000),
    ):
        """Validation step after epoch end"""
        self.set_mode("eval")
        model.cuda()

        loader = enumerate(val_loader)

        loss_list, pred_list, gt_list = [], [], []
        for i, (im_ids, im, mask) in loader:
            n = len(im_ids)
            if n < 1:
                continue

            im = im.cuda()
            mask = mask.cuda()

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logit = model(im)
                    # loss = model.loss(self.criterion, logit, mask).cpu().numpy()
                    loss = np.zeros([])

            mask = mask.cpu().numpy().astype(np.uint8, copy=False)
            if self.task == "seg":
                pred = torch.argmax(logit[0], 1).cpu().numpy().astype(np.uint8, copy=False)
            else:
                pred = torch.sigmoid(logit[0]).data.cpu().numpy()

            pred_list.extend(pred)
            gt_list.extend(mask)
            loss_list.extend(np.atleast_1d(loss))

        pred_list = np.array(pred_list)
        gt_list = np.array(gt_list)
        if self.task == "seg":
            acc = metrics.cmp_balanced_cls_acc(pred_list, gt_list, model.num_classes, mean=True)
            dice = metrics.cmp_pos_dice(pred_list, gt_list, model.num_classes, iou_instead=True)
        else:
            acc = metrics.cmp_binary_acc(pred_list, gt_list, mean=True)
            dice = acc

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(val_loader.batch_size, 3, figsize=(20, 40))
            # for i in range(val_loader.batch_size):
            #     axs[i, 0].imshow(im.numpy()[i, ...].squeeze(), cmap=plt.cm.bone)
            #     axs[i, 0].imshow(mask.numpy()[i, ...].squeeze(), alpha=0.3)
            #     axs[i, 1].imshow(torch.sigmoid(logit).cpu().numpy()[i, ...].squeeze())
            #     axs[i, 2].imshow(np.sum(
            #         [(j + 1) * (255 / len(pred[i])) * pred[i][j] / 255 for j in range(len(pred[i]))], axis=0).squeeze())
            # plt.show()

        metrics_out = dict(sum=dice + acc, dice=dice, acc=acc, loss=np.mean(loss_list))

        # Append epoch data to metrics dict
        for metric, value in metrics_out.items():
            H.update_log(self.val_log, metric, value)

        self.print_metrics()
        return metrics_out

    ######################## HELPER FUNS ##########################

    def predict(self, test_loader, pred_zip=None, pbar=False, ths=0.5):
        self.set_mode("valid")
        self.net.cuda()

        n_images = len(test_loader.dataset)
        n_iter = n_images / test_loader.batch_size
        if pbar:
            loader = tqdm(enumerate(test_loader), total=n_iter)
        else:
            loader = enumerate(test_loader)

        index_vec = []

        count = 0
        for i, (im_ids, im, mask) in loader:
            n = len(im_ids)
            if n < 1:
                continue
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(test_loader.batch_size // 4, 4)
            # axs = axs.ravel()
            # for i in range(len(index)):
            #     axs[i].imshow(np.moveaxis(im.numpy()[i, ...], 0, -1))
            # plt.show()

            with torch.no_grad():
                im = im.cuda()
                mask = mask.numpy()
                logits = self.net(im)[0]
                if self.task == "seg":
                    pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.float16, copy=False)
                if self.task == "cls":
                    pred = (
                        torch.sigmoid(logits)
                        .squeeze()
                        .cpu()
                        .numpy()
                        .astype(np.float16, copy=False)
                    )
                    pred = pred > ths

            if pred_zip is not None:
                for z, im_id in enumerate(im_ids):
                    pred_fn = im_id + ".png"
                    yp_img = np.uint8(pred[z] * 255)
                    img_bytes = cv2.imencode(".png", yp_img)[1].tobytes()
                    with ZipFile(pred_zip, "a") as f:
                        f.writestr(pred_fn, img_bytes)

            if not i:
                shape = (n_images, *pred.shape[1:])
                pred_vec = np.zeros(shape, dtype=np.float16)
                mask_vec = np.zeros((n_images, *mask.shape[-2:]), dtype=np.float16)

            pred_vec[count : count + n] = pred
            mask_vec[count : count + n] = mask
            index_vec.extend(im_ids)

            count += n

        return index_vec, pred_vec, mask_vec

    def print_metrics(self):
        msg = ""
        for metric, value in self.train_log.items():
            msg += "train {:}: {:.3f}\t".format(metric, value[-1])
        for metric, value in self.val_log.items():
            msg += "val {:}: {:.3f}\t".format(metric, value[-1])
        print(msg)

    def set_encoder_trainable(self, state):
        parameters = self.net.encoder.parameters()
        # parameters = self.net[0].parameters()
        if state:
            for param in parameters:
                param.requires_grad = True
        else:
            for param in parameters:
                param.requires_grad = False

        self.freeze_encoder = not state

    def set_mode(self, mode):
        self.mode = mode
        if mode in ["eval", "valid", "test"]:
            self.net.eval()
            self.is_training = False
        elif mode in ["train"]:
            self.is_training = True
            self.net.train()
            if self.freeze_encoder:
                self.net.encoder.apply(H.set_batchnorm_eval)
            if self.freeze_bn:
                self.net.apply(H.set_batchnorm_eval)
        else:
            raise NotImplementedError

    ####################### I/O FUNS ##############################
    def update_tbX(self, step, optimizer):
        """Update SummaryWriter for tensorboard"""
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar("lr_group{}".format(i), param_group["lr"], step)
        for tk, value in self.train_log.items():
            self.writer.add_scalar("train_{}".format(tk), value[-1], step)
        for vk, value in self.val_log.items():
            self.writer.add_scalar("val_{}".format(vk), value[-1], step)

    def save_best_model(self, model, metric):
        if (self.val_mode == "max" and metric > self.best_metric) or (
            self.val_mode == "min" and metric < self.best_metric
        ):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ":".join(str(datetime.datetime.now()).split(":")[:2])
            if self.fold is not None:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    "{:}_Fold{:}_Epoch{}_val{:.3f}".format(date, self.fold, self.epoch, metric),
                )
            else:
                self.best_model_path = os.path.join(
                    self.save_dir, "{:}_Epoch{}_val{:.3f}".format(date, self.epoch, metric)
                )

            torch.save(model.state_dict(), self.best_model_path)

    def save_training_log(self):
        d = dict()
        for tk in self.train_log.keys():
            d["train_{}".format(tk)] = self.train_log[tk]
        for vk in self.val_log.keys():
            d["val_{}".format(vk)] = self.val_log[vk]

        df = pd.DataFrame(d)
        df.index += 1
        df.index.name = "Epoch"

        date = ":".join(str(datetime.datetime.now()).split(":")[:2])
        if self.fold is not None:
            p = os.path.join(self.save_dir, "{:}_Fold{:}_TrainLog.csv".format(date, self.fold))
        else:
            p = os.path.join(self.save_dir, "{:}_TrainLog.csv".format(date))

        df.to_csv(p, sep=";")

        with open(p, "a") as fd:
            fd.write(self.comment)

    def update_ema(self, decay=0.99):
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(self.net)
            self.ema_model.cuda()

        else:
            par1 = self.ema_model.state_dict()
            par2 = self.net.state_dict()
            with torch.no_grad():
                for k in par1.keys():
                    par1[k].data.copy_(par1[k].data * decay + par2[k].data * (1 - decay))

        # # Save EMA model
        # date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
        # if self.fold is not None:
        #     path = os.path.join(self.save_dir,
        #                         '{:}_Fold{:}_Epoch{:}_ema'.format(date, self.fold, self.epoch))
        # else:
        #     path = os.path.join(self.save_dir,
        #                         '{:}_Epoch{:}_ema'.format(date, self.epoch))
        #
        # torch.save(self.ema_model.state_dict(), path)

    def load_model(self, path=None, best_model=False):
        if best_model:
            self.net.load_state_dict(torch.load(self.best_model_path))
        else:
            sd = torch.load(path)
            try:
                state_dict = sd["state_dict"]
            except KeyError:
                state_dict = sd

            # del state_dict['loss.weight'] #TODO: REMOVE THIS

            self.net.load_state_dict(state_dict, strict=True)
            # self.net.load_state_dict(torch.load(path)['state_dict'], strict=True)
        print("Model checkpoint loaded from: {}".format(path))

    def create_save_folder(self):
        cwd = os.getcwd()
        name = type(self.net).__name__
        fold = "Fold{}".format(self.fold)
        exp_id = time.strftime("%d%h_%H:%M")
        self.save_dir = os.path.join(cwd, "Saves", name, fold, exp_id)
        # Create dirs recursively if does not exist
        _dir = ""
        for d in self.save_dir.split("/"):
            if d == "":
                continue
            _dir += "/" + d
            if not os.path.exists(_dir):
                os.makedirs(_dir)
