import argparse
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from datasets import WHUBuilding, MassachuseetsBuilding
from model import spsnet
from utils.seg_metric import SegmentationMetric
from model import spsnet, UNetV2, UNet, BRRNet, deeplabv3plus, SegNet

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


def main():
    if args.model_name == "SPSNet":
        model = spsnet.__dict__["SPSNet"](
            in_channel=args.in_channel, out_channel=args.out_channel, spc=args.spc
        ).to(device)
    elif args.model_name == "UMobileNet":
        model = spsnet.__dict__["UMobileNet"](in_channel=args.in_channel, out_channel=args.out_channel).to(
            device
        )
    elif args.model_name == "UNetV2":
        # 预训练模型
        pretrained_path = r"weights/pvt_v2_b2.pth"
        model = UNetV2(
            n_classes=args.out_channel,
            deep_supervision=False,
            pretrained_path=pretrained_path,
        ).to(device)
    elif args.model_name == "UNet":
        model = UNet(n_channels=args.in_channel, n_classes=args.out_channel).to(device)
    elif args.model_name == "BRRNet":
        model = BRRNet(in_channels=args.in_channel, out_channels=args.out_channel).to(device)
    elif args.model_name == "Deeplabv3plus":
        model = deeplabv3plus.modeling.__dict__["deeplabv3plus_mobilenet"](
            num_classes=1, output_stride=16
        ).to(device)
    elif args.model_name == "SegNet":
        model = SegNet(in_channel=args.in_channel, out_channel=args.out_channel).to(device)
    else:
        print(f"不支持该模型：{args.model_name}")

    # 加载预训练模型
    if args.pretrain and os.path.isfile(args.pretrain):
        print("=> loading checkpoint '{}'".format(args.pretrain))
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint["epoch"]))
        # 删除checkpoint
        del checkpoint

    model.eval()
    # 准备数据集
    if args.dataset == "WHU_Building_Dataset":
        dataset_val = WHUBuilding(root=config.whu, mode=args.mode)
    elif args.dataset == "Massachusetts_Buildings_Dataset":
        dataset_val = MassachuseetsBuilding(root=config.massachuseets, version=args.version, mode=args.mode)
    val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=2, shuffle=False)

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
    label_name = [
        "acc",
        "iou_building",
        "f1_building",
        "pre_building",
        "rec_building",
        "miou",
        "val_loss",
    ]
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
    # 添加val_loss
    dict_index = dict(zip(label_name, index_list))
    print(dict_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    # network parameter
    parser.add_argument(
        "--model_name",
        type=str,
        default="UMobileNet",
        choices=[
            "SPSNet",
            "UMobileNet",
            "UNetV2",
            "UNet",
            "BRRNet",
            "Deeplabv3plus",
            "SegNet",
        ],
    )
    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="Massachusetts_Buildings_Dataset",
        choices=["Massachusetts_Buildings_Dataset", "WHU_Building_Dataset"],
    )
    parser.add_argument("--version", default="512_15", type=str, help="数据集的版本")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--mode", default="test", type=str)
    # 模型路径
    parser.add_argument(
        "--pretrain",
        default=r"checkpoints/umobilenet-whu/best_model.pt",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--in_channel", default=3, type=int, help="the number of input data channel")
    parser.add_argument("--out_channel", default=1, type=int, help="the number of output data channel")
    parser.add_argument("--spc", type=int, default=3, help="the channel of super pixel segment plug")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = config()
    main()
