import os

import cv2
import numpy as np
import torch
import pydicom


def read_dicom(path, color=True):
    dcm = pydicom.dcmread(path)
    # if dcm.file_meta.TransferSyntaxUID is None:
    #     dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    im = dcm.pixel_array
    if color:
        im = np.atleast_3d(im).repeat(3, 2)
    return im


def read_image(path, gray=False):
    assert os.path.exists(path), 'Path does not exist: {}'.format(path)
    if path.endswith('.dcm'):
        im = read_dicom(path, color=True)
    else:
        im = cv2.imread(path)
        if gray:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def read_multiclass_mask(list_of_rle, height=256, width=1600):
    mask = np.zeros((height, width), dtype=np.uint8)
    for i, rle in enumerate(list_of_rle):
        if rle == '-1':
            continue
        mask += rle2mask_steel(rle, height, width, value=(i + 1))
    return mask


def mixup_data(x, y, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def crop_image_mask(image, mask, tgt_size):
    if tgt_size is None or image.shape == tgt_size:
        return image, mask
    width = image.shape[1]
    tgt_width = tgt_size[1]

    x0 = np.random.randint(0, width - tgt_width)
    x1 = x0 + tgt_width

    return image[:, x0:x1], mask[:, x0:x1]


def resize_image_mask(image, mask, tgt_size, inter_type=cv2.INTER_AREA):
    if tgt_size is None or image.shape == tgt_size:
        return image, mask
    image = cv2.resize(image, (tgt_size[1], tgt_size[0]), interpolation=inter_type)
    mask = cv2.resize(mask, (tgt_size[1], tgt_size[0]), interpolation=inter_type)
    return image, mask


def uint2float(im):
    im = np.asarray(im) / 255.
    return im


def toTensor(im):
    im = np.moveaxis(np.atleast_3d(im), -1, 0)
    im = torch.from_numpy(im).float()
    return im


def default_batch_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return [], [], []

    im_ids, im, mask = [], [], []
    for b in batch:
        im_ids.append(b[0])
        im.append(b[1])
        mask.append(b[2])

    im = torch.stack(im, dim=0)
    mask = torch.stack(mask, dim=0)
    return im_ids, im, mask


def mask2rle(mask, n_classes=4):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n_classes + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append('')
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs


def rle2mask_steel(rle, height, width, value=255):
    mask = np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    for index, start in enumerate(starts):
        mask[start:start + lengths[index]] = value

    return mask.reshape(width, height).T


def rle2mask_ptx(rle, width, height, value=255):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = value
        current_position += lengths[index]

    return mask.reshape(width, height).T


def merge_binary_masks(masks, labels):
    mask = np.zeros_like(masks[0])
    for m, l in zip(masks, labels):
        mask[m > 0] = l
    return mask
