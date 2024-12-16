import argparse
import warnings
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from model import spsnet, UNetV2, UNet, BRRNet, deeplabv3plus, SegNet
from tools.whu_dataset_preprocess import save_colored_mask
import cv2

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


def feature_vis(feature, save_path):
    # feaats形状: [1,c,h,w] 类型：Tensor，batch_size=1
    channel_mean = torch.mean(feature, dim=1, keepdim=True)
    channel_mean = F.interpolate(channel_mean, size=(512, 512), mode="bilinear", align_corners=False)
    channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
    channel_mean = (
        ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255
    ).astype(np.uint8)
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    cv2.imwrite(str(save_path), channel_mean)


# hp: 超参数
def main():
    # 加载模型
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
            n_classes=args.out_channel, deep_supervision=False, pretrained_path=pretrained_path
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

    # 加载网络
    if args.model_path:
        if args.model_path.is_file():
            print("=> loading network '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no network found at '{}'".format(args.model_path))
        model.eval()

    # 加载图像
    img_list = list(args.img_path.glob("*.tif"))
    if len(img_list) == 0:
        print("Wrong data dir or suffix!")
        exit(1)
    print("{} samples found".format(len(img_list)))
    # 判断输出文件夹是否存在
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True)
    # 图像预处理
    input_transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    # 对每个image逐个预测
    with torch.no_grad():
        for img_path in tqdm(img_list[:10]):
            predict(model, img_path, args.output_path, input_transformer, vis_feature=args.vis_feature)


def predict(model, img_path, output_path, transformer=None, vis_feature=True):
    # 获取文件名
    img_name = img_path.stem
    # 对图像进行预处理
    if transformer is not None:
        img_data = transformer(np.asarray(Image.open(img_path))).to(device).unsqueeze(0)
    else:
        img_data = np.asarray(Image.open(img_path)).to(device).unsqueeze(0)
    # 输出结果并转化成标签
    output = model(img_data)
    if vis_feature:
        save_path = output_path.joinpath(f"{img_name}_{args.model_name.lower()}_vis.png")
        feature_vis(output, save_path)
    else:
        output = output.squeeze(0).squeeze(0).sigmoid()
        output = (output > 0.5).int().cpu().numpy().astype(np.uint8)
        # 输出路径
        img_output_path = output_path.joinpath(f"{img_name}_{args.model_name.lower()}.png")
        # 按照调色板模式输出预测图
        save_colored_mask(output, img_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument(
        "--model_name",
        type=str,
        default="SPSNet",
        choices=["SPSNet", "UMobileNet", "UNetV2", "UNet", "BRRNet", "Deeplabv3plus", "SegNet"],
    )
    # 模型路径
    parser.add_argument("--model_path", default=Path(r"checkpoints/test/test_checkpoint_0001epoch.pt"))
    parser.add_argument("--in_channel", default=3, type=int, help="the number of input data channel")
    parser.add_argument("--out_channel", default=1, type=int, help="the number of PlugUnetv2 data channel")
    parser.add_argument("--spc", type=int, default=3, help="the channel of super pixel segment plug")
    parser.add_argument("--img_path", default=Path(r"/mnt/d/dataset/WHU/test/image"))
    parser.add_argument("--output_path", default=Path(r"/mnt/d/dataset/building-example/whu-test"))
    # others
    parser.add_argument("--vis_feature", default=False, help="是否进行特征可视化")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()
