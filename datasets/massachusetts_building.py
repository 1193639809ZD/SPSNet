# @Author: error: git config user.name & please set dead value or install git
# @Date: 2024-07-20 16:43
# @LastEditTime: 2024-09-03 15:33
# @Description:

from PIL import Image
from torch.utils.data import Dataset

from transforms import seg_transforms as et


class MassachuseetsBuilding(Dataset):
    """Massachusetts Building 数据集"""

    def __init__(self, root, version, mode="train"):
        """
        Args:
            root (Path): root path of datasets
            version (str): 数据集裁剪的版本，目前有224_0、512_15两个版本
            mode (str): 'train', 'valid', or 'test'
        """
        self.dataset_path = root.joinpath(version, mode)
        self.image_list = list(self.dataset_path.glob("image/*.tiff"))
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
        return len(self.image_list)

    def __getitem__(self, idx):
        # 一定要用image名找mask文件，不要用需要获取，如果文件不匹配，后续很难排查错误
        image_path = self.image_list[idx]
        mask_path = image_path.parent.parent.joinpath("label", image_path.stem + ".tif")
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # 数据预处理
        image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}
