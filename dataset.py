import cv2
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import dataset, Chain, training, optimizers, \
    iterators, reporter, cuda, serializers
import argparse

if cuda.available:
    xp = cuda.cupy
else:
    xp = np


class DataSet(dataset.DatasetMixin):
    def __init__(self, mode, size, width, height):
        self.size = size
        self.width = width
        self.height = height
        self.mode = mode

    def __len__(self):
        return self.size

    def get_example(self, _):
        img = np.array(np.random.rand(
            self.height, self.width, 3) * 255, np.uint8)
        if self.mode == "Lab2RGB":
            return xp.array([np.ndarray.flatten(
                cv2.cvtColor(img, cv2.COLOR_BGR2LAB) / 255)])\
                .astype('float32'),\
                xp.array(np.ndarray.flatten(img / 255)).astype('float32')
        if self.mode == "RGB2Lab":
            return xp.array([np.ndarray.flatten(img / 255)])\
                .astype('float32'),\
                xp.ndarray.flatten(cv2.cvtColor(
                    img, cv2.COLOR_BGR2LAB) / 255).astype('float32')
