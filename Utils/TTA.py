import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F


class TTAOp:
    def __call__(self, model, batch):
        with torch.no_grad():
            forwarded = torch.from_numpy(self.forward(batch.numpy())).cuda()
            return self.backward(self.to_numpy(model.forward(forwarded)))

    def forward(self, img):
        raise NotImplementedError

    def backward(self, img):
        raise NotImplementedError

    def to_numpy(self, batch):
        if isinstance(batch, list):
            return [x.data.cpu().numpy() for x in batch]
        else:
            return batch.data.cpu().numpy()


class BasicTTAOp(TTAOp):
    @staticmethod
    def op(img):
        raise NotImplementedError

    def forward(self, img):
        return self.op(img)

    def backward(self, img):
        return img[0]


class HMosaic(TTAOp):
    def __init__(self, step=200, n_class=5, **kwargs):
        super().__init__(**kwargs)
        self.n_class = n_class
        self.step = step

    def forward(self, img):
        self.B, _, self.H, self.W = img.shape
        extra = (self.H - self.step) // 2
        img = np.pad(img, [[0, 0], [0, 0], [0, 0], [extra, extra]], mode="symmetric")
        n = self.W // self.step
        out = []
        start = 0
        for i in range(n):
            stop = start + self.H
            out.append(img[:, :, :, start:stop])
            start += self.step
        out = np.stack(out, 0)
        return out.reshape(-1, 3, self.H, self.H)

    def backward(self, pred):
        extra = self.H - self.step
        pred = pred.reshape(-1, self.B, self.n_class, self.H, self.H)
        out = np.empty((self.B, self.n_class, self.H, self.W + extra)) * np.nan
        n = self.W // self.step
        start = 0
        for i in range(n):
            stop = start + self.H
            out[..., start:stop] = np.nanmean([out[..., start:stop], pred[i]], axis=0)
            start += self.step
        out = out[:, :, :, extra // 2 : -(extra // 2)]
        return out


class Nothing(BasicTTAOp):
    @staticmethod
    def op(img):
        return img


class HFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=2))


class VFlip(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(np.flip(img, axis=3))


class Transpose(BasicTTAOp):
    @staticmethod
    def op(img):
        return np.ascontiguousarray(img.transpose(0, 1, 3, 2))


class ScaleTTA(TTAOp):
    def __init__(self, scale, **kwargs):
        super(ScaleTTA, self).__init__(**kwargs)
        self.scale = scale

    def __call__(self, model, batch):
        with torch.no_grad():
            self.im_size = batch.shape[-2:]
            forwarded = self.forward(batch).cuda()
            return self.to_numpy([self.backward(model.forward(forwarded)[0])])

    @staticmethod
    def op(img, scale):
        return F.upsample(img, scale_factor=scale, mode="bilinear", align_corners=True)

    def forward(self, img):
        return self.op(img, self.scale)

    def backward(self, img):
        return F.upsample(img, size=self.im_size, mode="bilinear", align_corners=True)


class ResizeTTA(TTAOp):
    def __init__(self, tgt_size, pad, **kwargs):
        super(ResizeTTA, self).__init__(**kwargs)
        self.tgt_size = tgt_size
        self.pad = pad

    def __call__(self, model, batch):
        with torch.no_grad():
            self.im_size = batch.shape[-2:]
            forwarded = self.forward(batch).cuda()
            return self.to_numpy(self.backward(model.forward(forwarded)))

    @staticmethod
    def op(img, tgt_size, pad):
        if pad:
            H, W = img.shape[2:]
            _max = max(H, W)
            padH = _max - H
            padW = _max - W
            img = np.pad(
                img,
                [(0, 0), (0, 0), (padH // 2, padH // 2), (padW // 2, padW // 2)],
                mode="constant",
            )

        return F.upsample(img, size=(tgt_size, tgt_size), mode="bilinear", align_corners=True)

    def forward(self, img):
        return self.op(img, self.tgt_size, self.pad)

    def backward(self, img):
        return img


def chain_op(data, operations):
    for op in operations:
        data = op.op(data)
    return data


class ChainedTTA(TTAOp):
    @property
    def operations(self):
        raise NotImplementedError

    def forward(self, img):
        return chain_op(img, self.operations)

    def backward(self, img):
        return chain_op(img, reversed(self.operations))


class HVFlip(ChainedTTA):
    @property
    def operations(self):
        return [HFlip, VFlip]


class TransposeHFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip]


class TransposeVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, VFlip]


class TransposeHVFlip(ChainedTTA):
    @property
    def operations(self):
        return [Transpose, HFlip, VFlip]


# TODO: Fix and test ScaleVFlip TTA
class ScaleVFlip(ChainedTTA):
    def __init__(self, scale, **kwargs):
        super(ScaleVFlip, self).__init__(**kwargs)
        self.scale = scale

    @property
    def operations(self):
        return [VFlip, ScaleTTA]


if __name__ == "__main__":
    # transforms = [Nothing, HFlip, VFlip, Transpose, HVFlip, TransposeHFlip, TransposeVFlip, TransposeHVFlip, DoubleResize]
    transforms = [DoubleResize]

    class Model(object):
        def predict(self, x):
            return x

    root = "./Data/Train/images/"
    imgs = os.listdir(root)[:2]
    imgs = [cv2.imread(os.path.join(root, im)) / 255.0 for im in imgs]
    data = torch.from_numpy(np.moveaxis(np.stack((imgs)).astype(np.float32), -1, 1))
    model = Model()
    for cls in transforms:
        TTA = cls(mode=None)
        ret = TTA(model, data)
    assert np.allclose(ret, data)
