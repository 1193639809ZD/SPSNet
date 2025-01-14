"""
author:        eveleaf <eveleaf@outlook.com.ar>
date:          2024-12-16 15:37:03
lastModified:  2024-12-16 15:37:04
"""

# @Author: error: git config user.name & please set dead value or install git
# @Date: 2024-07-20 16:43
# @LastEditTime: 2024-09-03 15:33
# @Description:

from PIL import Image
from torch.utils.data import Dataset

from transforms import seg_transforms as et
import json

from config import config


class SegDataset(Dataset):
    """Massachusetts数据集"""

    def __init__(self, root, mode="train"):
        """
        Args:
            root (Path): 子数据集的根目录
            metadata_path (Path): path to metadata.json
            mode (str): 'train', 'valid', or 'test'
        """
        self.dataset_path = root
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be 'train' or 'test'.")

        with open(root.joinpath("metadata.json"), "r") as json_file:
            metadata = json.load(json_file)

        # 统一改成使用文件加载数据，不容易出错
        self.dataset_items = metadata.get(f"{mode}_items", [])

        train_transform = et.ExtCompose(
            [
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transform = et.ExtCompose(
            [
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if mode == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        # 一定要用image名找mask文件，不要用需要获取，如果文件不匹配，后续很难排查错误
        item = self.dataset_items[idx]
        image = Image.open(config.datasets.joinpath(item["image"]))
        label = Image.open(config.datasets.joinpath(item["label"]))
        # 数据预处理
        image, label = self.transform(image, label)

        return {"image": image, "label": label}
