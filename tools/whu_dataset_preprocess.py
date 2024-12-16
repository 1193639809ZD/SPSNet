from pathlib import Path

import imgviz
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image(file_path):
    img = Image.open(file_path)
    data = np.asarray(img, dtype="int32")
    return data


def save_colored_mask(mask, save_path):
    mask = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    mask.putpalette(colormap.flatten())
    mask.save(save_path)


if __name__ == '__main__':
    # 数据集根路径
    dataset_root = Path(r'D:\dataset\whu_dataset_building')

    train_root = dataset_root.joinpath('train')
    train_label = train_root.joinpath('label')

    test_root = dataset_root.joinpath('test')
    test_label = test_root.joinpath('label')

    val_root = dataset_root.joinpath('val')
    val_label = val_root.joinpath('label')

    train_mask_list = list(train_label.glob('*'))
    for mask in tqdm(train_mask_list):
        mask_data = load_image(mask)
        mask_data[mask_data == 255] = 1
        save_colored_mask(mask_data, mask)

    test_mask_list = list(test_label.glob('*'))
    for mask in tqdm(test_mask_list):
        mask_data = load_image(mask)
        mask_data[mask_data == 255] = 1
        save_colored_mask(mask_data, mask)

    val_mask_list = list(val_label.glob('*'))
    for mask in tqdm(val_mask_list):
        mask_data = load_image(mask)
        mask_data[mask_data == 255] = 1
        save_colored_mask(mask_data, mask)
