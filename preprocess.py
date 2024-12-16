import glob
import os

import numpy as np
import tqdm
from PIL import Image

from config import Config


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    if img.mode == 'P':
        img.convert('RGB')
    data = np.asarray(img, dtype="int32")
    return data


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
    img_id = os.path.basename(mask_path).split(".")[0]  # 图像名称
    mask = load_image(mask_path)
    img_path = mask_path.replace("output", "input")
    img_path = img_path.replace("tif", "tiff")
    img = load_image(img_path)

    count = 0
    num_skipped = 1
    for i in Y_points:
        for j in X_points:
            new_image = img[i:i + split_height, j:j + split_width]
            new_mask = mask[i:i + split_height, j:j + split_width, 0]
            new_mask[new_mask > 0] = 255
            # Skip any Image that is more than 99% empty.
            if np.any(new_mask):
                num_black_pixels, num_white_pixels = np.unique(new_mask, return_counts=True)[1]

                if num_white_pixels / num_black_pixels < 0.01:
                    num_skipped += 1
                    continue

            mask_ = Image.fromarray(new_mask.astype(np.uint8))
            mask_.save("{}/{}_{}.jpg".format(mask_dir, img_id, count), "JPEG")
            im = Image.fromarray(new_image.astype(np.uint8))
            im.save("{}/{}_{}.jpg".format(image_dir, img_id, count), "JPEG")
            count = count + 1


if __name__ == '__main__':

    config = Config()

    train_dir = config.train
    valid_dir = config.valid
    # 0.14是重叠率
    X_points = start_points(config.IMAGE_SIZE, config.CROP_SIZE, 0.14)
    Y_points = start_points(config.IMAGE_SIZE, config.CROP_SIZE, 0.14)

    # Training data
    train_img_dir = os.path.join(train_dir, "input")
    train_mask_dir = os.path.join(train_dir, "output")
    train_img_crop_dir = os.path.join(config.train, "input_crop")
    os.makedirs(train_img_crop_dir, exist_ok=True)
    train_mask_crop_dir = os.path.join(config.train, "mask_crop")
    os.makedirs(train_mask_crop_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(train_img_dir, '**', '*.tiff'), recursive=True)
    mask_files = glob.glob(os.path.join(train_mask_dir, '**', '*.tif'), recursive=True)
    print("Length of image :", len(img_files))
    print("Length of mask :", len(mask_files))
    # assert len(img_files) == len(mask_files)

    for mask_path in tqdm.tqdm(mask_files, desc='Cropping Training images'):
        crop_image_mask(train_img_crop_dir, train_mask_crop_dir, mask_path, X_points, Y_points, config.CROP_SIZE,
                        config.CROP_SIZE)

    # Validation data
    valid_img_dir = os.path.join(config.valid, "input")
    valid_mask_dir = os.path.join(config.valid, "output")
    valid_img_crop_dir = os.path.join(config.valid, "input_crop")
    os.makedirs(valid_img_crop_dir, exist_ok=True)
    valid_mask_crop_dir = os.path.join(config.valid, "mask_crop")
    os.makedirs(valid_mask_crop_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(valid_img_dir, '**', '*.tiff'), recursive=True)
    mask_files = glob.glob(os.path.join(valid_mask_dir, '**', '*.tif'), recursive=True)
    assert len(img_files) == len(mask_files)

    for mask_path in tqdm.tqdm(mask_files, desc='Cropping Validation images'):
        crop_image_mask(valid_img_crop_dir, valid_mask_crop_dir, mask_path, X_points, Y_points, config.CROP_SIZE,
                        config.CROP_SIZE)
