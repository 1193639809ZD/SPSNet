import argparse
import os
import warnings

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from config import config
from datasets import WHUBuilding, MassachuseetsBuilding, YqTechDataset
from model import spsnet, UNetV2, UNet, BRRNet, deeplabv3plus, SegNet

from utils import metrics
from utils.DiceLoss import DiceLoss
from utils.EntropyLoss import EntropyLoss
from utils.seg_metric import SegmentationMetric

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


# config: 超参数
def main(name):
    checkpoint_dir = "{}/{}".format(args.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # os.makedirs("{}/{}".format(config.log, name), exist_ok=True)

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

    DLoss = DiceLoss()
    # BCELoss = torch.nn.BCELoss()
    CELoss = torch.nn.CrossEntropyLoss()
    ELoss = EntropyLoss()

    # 加载optimizer
    optimizer = optim.__dict__[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 加载scheduler
    lr_scheduler = optim.lr_scheduler.__dict__[args.lr_scheduler](
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # starting params
    best_miou = 0
    # 断点续训
    start_epoch = 0
    if args.pretrain and os.path.isfile(args.pretrain):
        print("=> loading checkpoint '{}'".format(args.pretrain))
        checkpoint = torch.load(args.pretrain)
        # start_epoch = checkpoint["epoch"]
        best_miou = checkpoint["best_miou"]
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint["epoch"]))
        # 删除checkpoint
        del checkpoint

    # 准备数据集
    if args.dataset == "WHU_Building_Dataset":
        dataset_train = WHUBuilding(root=config.whu, mode="train")
        dataset_val = WHUBuilding(root=config.whu, mode="test")
    elif args.dataset == "Massachusetts_Buildings_Dataset":
        dataset_train = MassachuseetsBuilding(root=config.massachuseets, version=args.version, mode="train")
        dataset_val = MassachuseetsBuilding(root=config.massachuseets, version=args.version, mode="test")
    elif args.dataset == "YqTech":
        # 子数据集：massachusetts、deepglobe、wayback
        dataset_train = YqTechDataset(root=config.yqtech, sub=args.yqsub, mode="train")
        dataset_val = YqTechDataset(root=config.yqtech, sub=args.yqsub, mode="test")
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=2, num_workers=2, shuffle=False)

    # start train
    for epoch in range(start_epoch, args.epochs):

        # step the learning rate scheduler
        lr_scheduler.step()
        train_dice = metrics.MetricTracker()
        # train_bce = metrics.MetricTracker()
        train_ce = metrics.MetricTracker()
        # train_e = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        # iterate over data
        loader = tqdm(train_dataloader, desc=f"training Epoch {epoch + 1}/{args.epochs}")
        for idx, data in enumerate(loader):

            # get the image and wrap in Variable
            image = data["image"].to(device)
            mask = data["mask"].to(device, torch.int64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # predict = model(image).sigmoid()
            predict = model(image).softmax(dim=1)
            # 计算损失
            # dice_score = DLoss(predict, mask.unsqueeze(dim=1))
            # bce_score = BCELoss(predict, mask.unsqueeze(dim=1).float())
            mask = F.one_hot(mask, num_classes=args.out_channel).permute(0, 3, 1, 2)
            dice_score = DLoss(predict, mask)
            ce_score = CELoss(predict, mask.float())

            # e_score = ELoss(predict)
            # loss = bce_score + dice_score + e_score
            # loss = ce_score + dice_score + e_score
            loss = ce_score + dice_score

            # backward
            loss.backward()
            optimizer.step()
            # 采用dice系数评价
            train_dice.update(dice_score.item(), predict.size(0))
            # train_bce.update(ce_score.item(), predict.size(0))
            train_ce.update(ce_score.item(), predict.size(0))
            # train_e.update(e_score.item(), predict.size(0))
            train_loss.update(loss.data.item(), predict.size(0))
            # 更新tqdm进度条
            if idx % args.logging_step == 0:
                loader.set_postfix(
                    **{
                        "lr": optimizer.param_groups[0]["lr"],
                        # "bce_score": "{:.4f}".format(train_bce.avg),
                        "ce_score": "{:.4f}".format(train_ce.avg),
                        "dice_score": "{:.4f}".format(train_dice.avg),
                        # "e_score": "{:.4f}".format(train_e.avg),
                        "Loss": "{:.4f}".format(train_loss.avg),
                    }
                )

        # 每两轮上传验证集指标以及保存
        if (epoch + 1) % args.validation_interval == 0:
            val_index = validation(val_dataloader, model)

            # 存储最好的模型
            if val_index["miou"] > best_miou:
                best_miou = val_index["miou"]
                save_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "arch": args.model_name,
                        "state_dict": model.state_dict(),
                        "best_miou": best_miou,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )

            # 存储每一轮模型
            save_path = os.path.join(checkpoint_dir, "%s_checkpoint_%04depoch.pt" % (name, epoch + 1))
            torch.save(
                {
                    "epoch": epoch,
                    "arch": args.model_name,
                    "state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            # 在val_index添加epoch信息，并上传到wandb云平台
            val_index["epoch"] = epoch + 1
            print(val_index)
            # 更新学习率参数
            print("Saved checkpoint to: %s" % save_path)


def validation(valid_loader, model):
    # 声明需要用到的评价矩阵
    metric = SegmentationMetric(args.out_channel, device)
    if args.out_channel == 1:
        BCELoss = torch.nn.BCELoss()
    else:
        CELoss = torch.nn.CrossEntropyLoss()
    val_loss = metrics.MetricTracker()

    # 切换模式
    model.eval()

    loader = tqdm(valid_loader, desc="validation")
    for idx, data in enumerate(loader):
        image = data["image"].to(device)
        mask = data["mask"].to(device, torch.int64)
        # 输出结果, 计算损失
        if args.out_channel == 1:
            predict = model(image).sigmoid()
            bce_score = BCELoss(predict, mask.unsqueeze(dim=1).float())
            val_loss.update(bce_score.item(), predict.size(0))
            # 计算混淆矩阵
            metric.add_batch((predict > 0.5).squeeze(dim=1).to(torch.int), mask)
        else:
            predict = model(image).softmax(dim=1)
            ce_score = CELoss(predict, mask)
            val_loss.update(ce_score.item(), predict.size(0))
            # 计算混淆矩阵
            metric.add_batch(predict.argmax(dim=1), mask)

        if idx % args.logging_step == 0:
            loader.set_postfix(**{"Loss": "{:.4f}".format(val_loss.avg)})

    # print(metric.confusion_matrix)
    print(metric.intersection_over_union().tolist())
    # 创建指标列表
    index_list = np.array([])
    # label_name = [
    #     "acc",
    #     "iou_building",
    #     "f1_building",
    #     "pre_building",
    #     "rec_building",
    #     "miou",
    #     "val_loss",
    # ]
    # # 添加acc
    # index_list = np.append(index_list, metric.pixel_accuracy().item())
    # # 添加class_iou
    # index_list = np.append(index_list, metric.intersection_over_union().tolist()[1])
    # # 添加f1-score
    # index_list = np.append(index_list, metric.f1_score().tolist()[1])
    # # 添加class_pre
    # index_list = np.append(index_list, metric.class_pixel_precision().tolist()[1])
    # # 添加class_rec
    # index_list = np.append(index_list, metric.class_pixel_recall().tolist()[1])
    # # 添加miou
    # index_list = np.append(index_list, metric.mean_intersection_over_union().item())
    # # 添加val_loss
    # index_list = np.append(index_list, val_loss.avg)

    label_name = [
        # "iou",
        "miou",
        "val_loss",
    ]
    # index_list = np.append(index_list, metric.intersection_over_union().tolist())
    index_list = np.append(index_list, metric.mean_intersection_over_union().item())
    index_list = np.append(index_list, val_loss.avg)
    dict_index = dict(zip(label_name, index_list))

    # 打开model的训练模式
    model.train()

    return dict_index


if __name__ == "__main__":
    # 获取命令行参数
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="checkpoint文件夹路径")
    parser.add_argument("--name", default="umobilenet-yq-deep", type=str, help="Experiment name")
    # 数据集设置
    parser.add_argument(
        "--dataset",
        type=str,
        default="YqTech",
        choices=["Massachusetts_Buildings_Dataset", "WHU_Building_Dataset", "YqTech"],
    )
    parser.add_argument("--version", default="512_15", type=str, help="数据集的版本")
    parser.add_argument(
        "--yqsub",
        default="deepglobe",
        type=str,
        choices=["massachusetts", "deepglobe", "wayback"],
        help="yqtech数据集的子数据集选择",
    )
    parser.add_argument("--batch_size", default=4, type=int)
    # model参数
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
    parser.add_argument(
        "--pretrain",
        default="checkpoints/umobilenet-yq-deep/best_model.pt",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--in_channel", default=3, type=int, help="the number of input data channel")
    parser.add_argument("--out_channel", default=10, type=int, help="the number of classes")
    parser.add_argument("--spc", type=int, default=3, help="the channel of super pixel segment plug")
    # optimizer参数
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["RMSprop", "Adam"])
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # scheduler参数
    parser.add_argument("--lr_scheduler", type=str, default="StepLR")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    # 其他参数
    parser.add_argument(
        "--epochs",
        default=60,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--logging_step", type=int, default=20)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(name=args.name)
