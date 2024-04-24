# from __future__ import print_function, division
from logging import root

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path

from torchvision.datasets.folder import IMG_EXTENSIONS


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
            # ('../data/usps2mnist/images/mnist_train_image/0.jpg',array([5]))
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images  # ('../data/usps2mnist/images/mnist_train_image/1.jpg', array([0]))


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')  # <PIL.Image.Image image mode=RGB size=28x28 at 0x1E6020D0630>


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader  # 使用路径将图片加载为PIL格式

    def __getitem__(self, index):
        path, target = self.imgs[index]  # ('../data/usps2mnist/images/mnist_train_image/0.jpg',array([5]))
        # image-clef
        path = path.rsplit('/', 1)[0] + f'/{target}/' + path.rsplit('/', 1)[1]
        img = self.loader(path)  # <PIL.Image.Image image mode=L size=28x28 at 0x21628703668>
        if self.transform is not None:
            img = self.transform(img)  # torch.Size([3, 28, 28]) # 因为mnist数据集所占用的内存较大，更倾向于在调用getitem函数的时候进行图片的转化。
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target  # {tensor:(1,28,28)}   array([5])

    def __len__(self):
        return len(self.imgs)  # 60000


class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
