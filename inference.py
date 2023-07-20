import argparse
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from datasets.whu_building import WHUBuilding
from model import spsnet
from utils.seg_metric import SegmentationMetric

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


def main():
    if args.model_name == 'SPSNet':
        model = spsnet.__dict__[args.model_name](in_channel=args.in_channel, out_channel=args.out_channel,
                                                 spc=args.spc).to(device)
    else:
        model = spsnet.__dict__[args.model_name](in_channel=args.in_channel, out_channel=args.out_channel).to(device)
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading network '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            del checkpoint

        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

    model.eval()
    datasets = WHUBuilding(root=config.whu, mode='test')
    val_dataloader = DataLoader(datasets, batch_size=1, num_workers=4)

    metric = SegmentationMetric(2, device)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_dataloader, desc="validation")):
            image = data["image"].to(device)
            mask = data["mask"].to(device)

            predict = model(image).sigmoid()
            # 计算混淆矩阵
            metric.add_batch((predict > 0.5).squeeze(dim=1).to(torch.int), mask)

    # 创建指标列表
    index_list = np.array([])
    label_name = ['acc', 'iou_building', 'f1_building', 'pre_building', 'rec_building', 'miou', 'val_loss']
    # 添加acc
    index_list = np.append(index_list, metric.pixel_accuracy().item())
    # 添加class_iou
    index_list = np.append(index_list, metric.intersection_over_union().tolist()[1])
    # 添加f1-score
    index_list = np.append(index_list, metric.f1_score().tolist()[1])
    # 添加class_pre
    index_list = np.append(index_list, metric.class_pixel_precision().tolist()[1])
    # 添加class_rec
    index_list = np.append(index_list, metric.class_pixel_recall().tolist()[1])
    # 添加miou
    index_list = np.append(index_list, metric.mean_intersection_over_union().item())

    dict_index = dict(zip(label_name, index_list))
    print(dict_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    # dataset parameter
    parser.add_argument('--mode', type=str, default='test')
    # network parameter
    parser.add_argument("--model_name", type=str, default='SPSNet')
    # 模型路径
    parser.add_argument("--model_path", default=r"")
    parser.add_argument('--in_channel', default=3, type=int, help="the number of input data channel")
    parser.add_argument('--out_channel', default=1, type=int, help="the number of output data channel")
    parser.add_argument('--spc', type=int, default=3, help="the channel of super pixel segment plug")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = config()
    main()
