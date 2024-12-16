# @Author: error: git config user.name & please set dead value or install git
# @Date: 2024-07-20 16:43
# @LastEditTime: 2024-09-03 15:43
# @Description:

from PIL import Image
from torch.utils.data import Dataset

from transforms import seg_transforms as et


class WHUBuilding(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, root, mode="train"):
        """
        Args:
            root (Path): root path of datasets
            mode (str): 'train', 'valid', or 'test'
        """
        self.dataset_path = root.joinpath(mode)
        self.image_list = list(self.dataset_path.glob("image/*.tif"))
        self.mask_list = list(self.dataset_path.glob("label/*.tif"))
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
        return len(self.mask_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        mask = Image.open(self.mask_list[idx])
        # 数据预处理
        image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}
