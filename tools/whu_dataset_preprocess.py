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
    dataset_root = Path(r'./data/WHU_Building_Dataset')

    train_root = dataset_root.joinpath('train')
    train_label = train_root.joinpath('label')
    train_mask = train_root.joinpath('mask')
    train_mask.mkdir(parents=True, exist_ok=True)

    test_root = dataset_root.joinpath('test')
    test_label = test_root.joinpath('label')
    test_mask = test_root.joinpath('mask')
    test_mask.mkdir(parents=True, exist_ok=True)

    val_root = dataset_root.joinpath('val')
    val_label = val_root.joinpath('label')
    val_mask = val_root.joinpath('mask')
    val_mask.mkdir(parents=True, exist_ok=True)

    train_mask_list = list(train_label.glob('*'))
    for mask in tqdm(train_mask_list):
        mask_data = load_image(mask)
        save_path = train_mask.joinpath(mask.stem + '.png')
        save_colored_mask(mask_data, save_path)

    test_mask_list = list(test_label.glob('*'))
    for mask in tqdm(test_mask_list):
        mask_data = load_image(mask)
        save_path = test_mask.joinpath(mask.stem + '.png')
        save_colored_mask(mask_data, save_path)

    val_mask_list = list(val_label.glob('*'))
    for mask in tqdm(val_mask_list):
        mask_data = load_image(mask)
        save_path = val_mask.joinpath(mask.stem + '.png')
        save_colored_mask(mask_data, save_path)
