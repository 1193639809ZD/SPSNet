import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from model import spsnet, UNetV2, UNet, BRRNet
from tools.whu_dataset_preprocess import save_colored_mask
from torchvision.models.feature_extraction import create_feature_extractor

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


# hp: 超参数
def main():
    # 加载模型
    if args.model_name == 'SPSNet':
        model = spsnet.__dict__[args.model_name](in_channel=args.in_channel, out_channel=args.out_channel, spc=args.spc).to(device)
    elif args.model_name == 'UNetV2':
        # 预训练模型
        pretrained_path = r"weights/pvt_v2_b2.pth"
        model = UNetV2(n_classes=args.out_channel, deep_supervision=False, pretrained_path=pretrained_path).to(device)
    elif args.model_name == "UNet":
        model = UNet(n_channels=args.in_channel, n_classes=args.out_channel).to(device)
    elif args.model_name == 'BRRNet':
        model = BRRNet(in_channels=args.in_channel, out_channels=args.out_channel).to(device)
    else:
        model = spsnet.__dict__[args.model_name](in_channel=args.in_channel, out_channel=args.out_channel).to(device)

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
    img_list = list(args.img_path.glob('*.tif'))
    if len(img_list) == 0:
        print('Wrong data dir or suffix!')
        exit(1)
    print('{} samples found'.format(len(img_list)))
    # 判断输出文件夹是否存在
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True)
    # 图像预处理
    input_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feature_extractor = create_feature_extractor(model, return_nodes={"outc": "output"})
    # 对每个image逐个预测
    with torch.no_grad():
        for img_path in tqdm(img_list):
            vis_feature(feature_extractor, img_path, args.output_path, input_transformer)


def vis_feature(feature_extractor, img_path, output_path, transformer=None):
    # 获取文件名
    img_name = img_path.stem
    # 对图像进行预处理
    if transformer is not None:
        img_data = transformer(np.asarray(Image.open(img_path))).to(device).unsqueeze(0)
    else:
        img_data = np.asarray(Image.open(img_path)).to(device).unsqueeze(0)
    # 输出特征
    output = feature_extractor(img_data)

    # 输出路径
    img_output_path = output_path.joinpath(f'{img_name}_{args.model_name.lower()}.png')
    # 按照调色板模式输出预测图
    image = Image.fromarray(output["output"][0].transpose(0,1).sum(1).detach().numpy().astype(np.uint8), mode="P")
    image.save(img_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument('--model_name', type=str, default='SPSNet', choices=['SPSNet', 'UMobileNet', 'UNetV2', 'UNet', 'BRRNet'])
    # 模型路径
    parser.add_argument('--model_path',
                        default=Path(r"checkpoints\SPS-UNet-whu\best_model.pt"))
    parser.add_argument('--in_channel', default=3, type=int, help="the number of input data channel")
    parser.add_argument('--out_channel', default=1, type=int, help="the number of PlugUnetv2 data channel")
    parser.add_argument('--spc', type=int, default=3, help="the channel of super pixel segment plug")
    parser.add_argument('--img_path', default=Path(r'D:/dataset/building-example/whu'))
    parser.add_argument('--output_path', default=Path(r'D:\dataset\building-example\whu-heatmap'))
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()
