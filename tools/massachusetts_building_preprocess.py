from PIL import Image
import numpy as np
from pathlib import Path
import imgviz
from tqdm import tqdm


def save_colored_mask(mask, save_path):
    mask = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    mask.putpalette(colormap.flatten())
    mask.save(save_path)


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def crop_image_mask(image_dir, mask_dir, mask_path, X_points, Y_points, split_height=224, split_width=224):
    # 加载图像和标签
    img_id = Path(mask_path).stem
    img = np.array(Image.open(label_file.parent.parent.joinpath("image", label_file.stem + ".tiff")))
    mask = np.array(Image.open(label_file))

    count = 0
    num_skipped = 1
    for i in Y_points:
        for j in X_points:
            new_image = img[i : i + split_height, j : j + split_width]
            new_mask = mask[i : i + split_height, j : j + split_width]
            # Skip any Image that is more than 99% empty.
            if np.any(new_mask):
                num_black_pixels, num_white_pixels = np.unique(new_mask, return_counts=True)[1]

                if num_white_pixels / num_black_pixels < 0.01:
                    num_skipped += 1
                    continue
            # 保存裁剪后图片
            im = Image.fromarray(new_image.astype(np.uint8))
            im.save("{}/{}_{}.tiff".format(image_dir, img_id, count))
            save_colored_mask(new_mask, "{}/{}_{}.tif".format(mask_dir, img_id, count))
            count = count + 1


dataset_root = Path(r"D:/dataset/Massachusetts-building")

sub_roots = list(dataset_root.iterdir())

# 1. 将label转化为单通道调色板模式，1表示建筑物，0表示背景
for sub_root in sub_roots:
    if not sub_root.is_dir():
        continue
    label_files = sub_root.glob("label/*.tif")
    for label_file in tqdm(label_files, ncols=200):
        lab_data = np.array(Image.open(label_file))
        new_lab_data = (lab_data == (255, 0, 0)).all(axis=2).astype(np.uint8)
        save_colored_mask(new_lab_data, label_file)

# 2. 将图片的空白区域的mask去除
for sub_root in sub_roots:
    if not sub_root.is_dir():
        continue
    label_files = sub_root.glob("label/*.tif")
    for label_file in tqdm(label_files, ncols=200):
        img_data = np.array(Image.open(label_file.parent.parent.joinpath("image", label_file.stem + ".tiff")))
        lab_data = np.array(Image.open(label_file))
        lab_data[(img_data == (255, 255, 255)).any(axis=2)] = 0
        save_colored_mask(lab_data, label_file)

# 3. 图片裁剪
process_root = Path(r"D:/dataset/Massachusetts-building-process")
X_points = start_points(1500, 256, 0.15)
Y_points = start_points(1500, 256, 0.15)
for sub_root in sub_roots:
    if not sub_root.is_dir():
        continue
    process_sub_root = process_root.joinpath(sub_root.name)
    image_dir = process_sub_root.joinpath("image")
    mask_dir = process_sub_root.joinpath("label")
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    label_files = sub_root.glob("label/*.tif")
    for label_file in tqdm(label_files):
        crop_image_mask(
            image_dir, mask_dir, label_file, X_points, Y_points, split_height=224, split_width=224
        )
