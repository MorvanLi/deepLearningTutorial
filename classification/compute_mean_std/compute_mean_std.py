# -*- coding: utf-8 -*-
"""
@Author  : Morvan Li
@FileName: compute_mean_std.py
@Software: PyCharm
@Time    : 8/15/22 5:41 PM
"""
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Tuple
import numpy as np

data_transform = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
}


# method 1
def getStat(train_data: str) -> Tuple[np.array, np.array]:
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print("Compute mean and variance for training data.")
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


# method 2
def getStat2(img_dir=str):
    img_channels = 3
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    img_name_list = [os.path.join(root, name) for root, dirs, files in os.walk(img_dir) for name in files]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_path in img_name_list:
        img = np.array(Image.open(img_path)) / 255.
        # for d in range(3):
        #     cumulative_mean[d] += img[:, :, d].mean()
        #     cumulative_std[d] += img[:, :, d].std()
        cumulative_mean += img.mean(axis=(0, 1))
        cumulative_std += img.std(axis=(0, 1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == "__main__":

    train_dataset = ImageFolder(
        root="../../flower_data", transform=data_transform["train"]
    )
    print(getStat(train_dataset))
    getStat2("../../flower_data")
