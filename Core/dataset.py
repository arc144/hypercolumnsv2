import os

import cv2
import numpy as np
import pandas as pd
import torch
import Utils.data_helpers as H
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler


class CocoABC(Dataset):
    """This is a class that inherits the Dataset class from PyTorch."""

    nb_classes = 81 + 1

    def __init__(self, df, root, remove_bg_only=False, transform=None, tgt_size=None):
        self.coco_obj = COCO(df)
        self.root = root

        self.catIds = []
        self.labels_map = dict(zip(range(1, BikeCar.nb_classes), range(1, BikeCar.nb_classes)))

        self.remove_bg_only = remove_bg_only
        self.tgt_size = tgt_size
        self.transform = transform

    def generate_meta(self):
        imgIds = self.coco_obj.getImgIds()
        # Set a df for DataSampler
        fgIds = list(
            np.concatenate([self.coco_obj.getImgIds(catIds=[cat]) for cat in self.catIds])
        )
        self.df = pd.DataFrame({"ImageId": imgIds, "FG": [False] * len(imgIds)}).set_index(
            "ImageId"
        )
        self.df.loc[fgIds, "FG"] = True

        if self.remove_bg_only:
            self.df = self.df[self.df["FG"] == True]

        self.size = len(self.df)
        self.ids = np.unique(self.df.index.values)
        self.mapping = dict(zip(self.ids, range(self.size)))
        print("Dataset comprised of {} images".format(self.size))
        print("{} pos labels found".format(self.df["FG"].values.mean()))

    def subsample_df(self, nb_images, seed=101):
        self.df = self.df.sample(nb_images, random_state=seed)
        self.generate_meta()

    def __len__(self):
        return self.size

    def apply_transform(self, im, mask):
        if self.transform is not None:
            mask = SegmentationMapsOnImage(mask, im.shape)
            im, mask = self.transform(image=im, segmentation_maps=mask)
            mask = mask.get_arr()

        return im, mask

    def get_image_path(self, image_id):
        filename = self.coco_obj.loadImgs([image_id])[0]["file_name"]
        return os.path.join(self.root, filename)

    def get_image_mask(self, image_id, height, width):
        annIds = self.coco_obj.getAnnIds(imgIds=[image_id], catIds=self.catIds)
        anns = self.coco_obj.loadAnns(annIds)
        if len(anns) == 0:
            return np.zeros((height, width), dtype=np.uint8)

        masks = [self.coco_obj.annToMask(ann) for ann in anns]
        labels = [self.labels_map[ann["category_id"]] for ann in anns]
        return H.merge_binary_masks(masks, labels)

    def load_image_gt(self, index):
        """Load image and gt from a subset dataframe"""
        image_id = self.ids[index]
        path = self.get_image_path(image_id)

        image = H.read_image(path)
        mask = self.get_image_mask(image_id, *image.shape[:2])

        return image_id, image, mask

    def __getitem__(self, index):
        # Load gts, images and process images
        image_id, image, mask = self.load_image_gt(index)
        if image is None:
            return image_id, None, None

        image, mask = H.resize_image_mask(image, mask, self.tgt_size)
        image, mask = self.apply_transform(image, mask)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(image)
        # axs[1].imshow(mask)
        # plt.show()

        image = H.uint2float(image)
        image = H.toTensor(image)
        mask = torch.from_numpy(mask).long()

        return image_id, image, mask


class BikeCar(CocoABC):
    nb_classes = 2 + 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catIds = [2, 3]
        self.labels_map = dict(zip(self.catIds, range(1, BikeCar.nb_classes)))
        self.generate_meta()


class PneumoDataset(Dataset):
    """This is a class that inherits the Dataset class from PyTorch."""

    nb_classes = 1 + 1

    def __init__(self, df, root, remove_bg_only=False, transform=None, tgt_size=None):
        self.df = df.set_index("ImageId")

        # Count the number of masks by image
        self.gb = self.df.groupby("ImageId")
        self.df["count"] = self.gb.transform("count")["EncodedPixels"]
        self.df.loc[self.df["EncodedPixels"] == "-1", "count"] = 0
        self.df["FG"] = self.df["count"] > 0

        if remove_bg_only:
            self.df = self.df[self.df["count"] > 0]

        self.tgt_size = tgt_size
        self.transform = transform
        self.generate_meta()

    def generate_meta(self):
        self.ids = np.unique(self.df.index.values)
        self.size = len(self.ids)
        self.mapping = dict(zip(self.ids, range(self.size)))
        print("Dataset comprised of {} images".format(self.size))
        print("{} pos labels found".format(self.df["FG"].values.mean()))

    def subsample_df(self, nb_images, seed=101):
        self.df = self.df.sample(nb_images, random_state=seed)
        self.generate_meta()

    def __len__(self):
        return self.size

    def apply_transform(self, im, mask):
        if self.transform is not None:
            mask = SegmentationMapsOnImage(mask.astype(bool), im.shape)
            im, mask = self.transform(image=im, segmentation_maps=mask)
            mask = mask.get_arr() * 255

        return im, mask

    def load_image_gt(self, index):
        """Load image and gt from a subset dataframe"""
        image_id = self.ids[index]

        path = self.df.loc[image_id, "Path"]
        image = H.read_image(path if isinstance(path, str) else path.values[0])
        h, w = image.shape[:2]

        if "EncodedPixels" in self.df.columns:
            rles = self.df.loc[image_id, "EncodedPixels"]
            if isinstance(rles, str) and rles == "-1":
                mask = np.zeros((w, h))
            else:
                if isinstance(rles, pd.Series):
                    rles = list(rles.values)

                elif isinstance(rles, str):
                    rles = [rles]

                mask = [H.rle2mask_ptx(rle, w, h) for rle in rles]
                mask = np.max(mask, axis=0)
        elif "Pneumothorax" in self.df.columns:
            mask = self.df.loc[image_id, "Pneumothorax"]
            mask = np.ones((h, w)) * 255 * mask
        else:
            mask = np.zeros((h, w))

        return image_id, image, mask

    def __getitem__(self, index):
        # Load gts, images and process images
        image_id, image, mask = self.load_image_gt(index)
        if image is None:
            return image_id, None, None

        image, mask = H.resize_image_mask(image, mask, self.tgt_size)
        mask = (mask > 127.5).astype(np.uint8) * 255

        image, mask = self.apply_transform(image, mask)

        image = H.uint2float(image)
        image = H.toTensor(image)
        mask = H.uint2float(mask)
        mask = H.toTensor(mask).long().squeeze(0)

        return image_id, image, mask


class BalanceClassSampler(Sampler):
    def __init__(
        self,
        dataset,
        pos_ratio_range=(0.20, 0.80),
        epochs=None,
        remove_bg_only=False,
        mode="mixed",
    ):
        self.epochs = epochs
        self.pos_ratio_range = pos_ratio_range
        self.remove_bg_only = remove_bg_only
        self.dataset = dataset
        self.mode = mode
        if mode not in ["mixed", "under", "over"]:
            raise ValueError(
                'Mode argument must be either "mixed", "over" or "under". Got {}'.format(mode)
            )
        self.current_epoch = 0

        if self.remove_bg_only:
            self.length = len(np.unique(self.dataset.df[self.dataset.df["FG"]].index.values))
        else:
            self._get_ids()
            self._get_length()

    def _get_ids(self):
        self.pos_ids = np.unique(self.dataset.df[self.dataset.df["FG"].values].index.values)
        self.neg_ids = np.unique(self.dataset.df[~self.dataset.df["FG"].values].index.values)
        self.pos_index = [self.dataset.mapping[id] for id in self.pos_ids]
        self.neg_index = [self.dataset.mapping[id] for id in self.neg_ids]

    def _get_length(self):
        if self.mode == "mixed":
            self.length = len(np.unique(self.dataset.df.index.values))
        elif self.mode == "under":
            self.length = 2 * min(len(self.pos_ids), len(self.neg_ids))
        elif self.mode == "over":
            self.length = 2 * max(len(self.pos_ids), len(self.neg_ids))

    def __iter__(self):
        if self.remove_bg_only:
            pos_ids = self.dataset.df[self.dataset.df["FG"].values].index.values
            pos_index = [self.dataset.mapping[id] for id in pos_ids]
            pos = np.random.choice(pos_index, len(pos_index), replace=False)
            return iter(pos)

        pos_num = int(self.length * self.positive_ratio) + 1
        neg_num = self.length - pos_num
        pos = np.random.choice(self.pos_index, pos_num, replace=True)
        neg = np.random.choice(self.neg_index, neg_num, replace=True)

        l = np.concatenate([pos, neg])
        np.random.shuffle(l)
        l = l[: self.length]
        return iter(l)

    @property
    def positive_ratio(self):
        min_ratio, max_ratio = self.pos_ratio_range
        return max_ratio - (max_ratio - min_ratio) / self.epochs * self.current_epoch

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __len__(self):
        return self.length


class DatasetFactory:
    def __init__(self, csv_path, root, dataset_name):
        self.csv_path = csv_path
        self.root = root
        self.is_coco = False
        if dataset_name == "ptx":
            self.data_fun = PneumoDataset
        elif dataset_name == "motocar":
            self.data_fun = BikeCar
            self.is_coco = True
        else:
            raise ValueError("Dataset name not recognized")

    def yield_loader(
        self,
        is_test,
        tgt_size,
        batch_size,
        aug=None,
        remove_bg_only=False,
        epochs=None,
        balanced_sampler=True,
        pos_ratio_range=(50, 50),
        sampling_mode="mixed",
        workers=12,
    ):
        """
        Proceed with PyTorch data pipeline.
        Return the dataloaders used in training
        """
        df = self.csv_path
        root = self.root
        if self.is_coco:
            if is_test:
                df = df.replace("instances_train", "instances_val")
                root = root.replace("train2017", "val2017")

        else:
            df = pd.read_csv(df)
            if is_test:
                if "Set" in df.columns:
                    df = df[df["Set"] == "val"]
            else:
                df = df[df["Set"] == "train"]

        dataset = self.data_fun(
            df,
            root,
            transform=None if is_test else aug,
            tgt_size=tgt_size,
            remove_bg_only=remove_bg_only,
        )
        if is_test:
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                num_workers=workers,
                batch_size=batch_size,
                collate_fn=H.default_batch_collate,
                pin_memory=True,
            )

        else:
            dataloader = DataLoader(
                dataset,
                num_workers=workers,
                collate_fn=H.default_batch_collate,
                pin_memory=True,
                sampler=BalanceClassSampler(
                    dataset,
                    pos_ratio_range=pos_ratio_range,
                    mode=sampling_mode,
                    epochs=epochs,
                    remove_bg_only=remove_bg_only,
                )
                if balanced_sampler
                else None,
                batch_size=batch_size,
                drop_last=True,
            )

        return dataloader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Utils import augmentations as Aug

    os.chdir("../")
    dataset = DatasetFactory(
        "/media/nvme/Datasets/COCO/annotations/instances_train2017.json",
        "/media/nvme/Datasets/COCO/train2017",
        "motocar",
    )
    dataloader = dataset.yield_loader(
        is_test=False,
        tgt_size=(256, 256),
        remove_bg_only=False,
        aug=Aug.Aug0,
        workers=0,
        balanced_sampler=False,
        epochs=30,
        batch_size=16,
    )
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
    for index, im, mask in dataloader:
        fig, axes = plt.subplots(4, 2)
        for i in range(4):
            im_i = np.moveaxis((im[i].numpy() * 255).astype(np.uint8), 0, -1)
            axes[i, 0].imshow(im_i)
            mask_i = mask[i].numpy().squeeze()
            axes[i, 1].imshow(mask_i)
        break

    plt.show()
