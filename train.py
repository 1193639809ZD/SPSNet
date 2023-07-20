import argparse
import os
import warnings

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from datasets.whu_building import WHUBuilding
from model import spsnet
from utils import metrics
from utils.DiceLoss import DiceLoss
from utils.EntropyLoss import EntropyLoss
from utils.seg_metric import SegmentationMetric

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


# config: 超参数
def main(name):
    checkpoint_dir = "{}/{}".format(config.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("{}/{}".format(config.log, name), exist_ok=True)
    if args.model_name == 'SPSNet':
        model = spsnet.__dict__[args.model_name](in_channel=args.in_channel, out_channel=args.out_channel,
                                                 spc=args.spc).to(device)
    else:
        model = spsnet.__dict__[args.model_name](in_channel=args.in_channel, out_channel=args.out_channel).to(device)
    DLoss = DiceLoss()
    BCELoss = torch.nn.BCELoss()
    ELoss = EntropyLoss()

    # 加载optimizer
    optimizer = optim.__dict__[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 加载scheduler
    lr_scheduler = optim.lr_scheduler.__dict__[args.lr_scheduler](optimizer, step_size=args.step_size, gamma=args.gamma)

    # starting params
    best_miou = 0
    # 断点续训
    start_epoch = 0
    if args.pretrain and os.path.isfile(args.pretrain):
        print("=> loading checkpoint '{}'".format(args.pretrain))
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint["epoch"]
        best_miou = checkpoint["best_miou"]
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint["epoch"]))
        # 删除checkpoint
        del checkpoint

    # 准备数据集
    dataset_train = WHUBuilding(root=config.whu, mode='train')
    dataset_val = WHUBuilding(root=config.whu, mode='test')
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=2, num_workers=2, shuffle=False)

    # start train
    for epoch in range(start_epoch, args.epochs):

        # step the learning rate scheduler
        lr_scheduler.step()
        train_dice = metrics.MetricTracker()
        train_bce = metrics.MetricTracker()
        train_e = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        # iterate over data
        loader = tqdm(train_dataloader, desc=f"training Epoch {epoch + 1}/{args.epochs}")
        for idx, data in enumerate(loader):

            # get the image and wrap in Variable
            image = data["image"].to(device)
            mask = data["mask"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            predict = model(image).sigmoid()

            # 计算损失
            dice_score = DLoss(predict, mask.unsqueeze(dim=1))
            bce_score = BCELoss(predict, mask.unsqueeze(dim=1).float())
            e_score = ELoss(predict)
            loss = bce_score + dice_score + e_score

            # backward
            loss.backward()
            optimizer.step()
            # 采用dice系数评价
            train_dice.update(dice_score.item(), predict.size(0))
            train_bce.update(bce_score.item(), predict.size(0))
            train_e.update(e_score.item(), predict.size(0))
            train_loss.update(loss.data.item(), predict.size(0))
            # 更新tqdm进度条
            if idx % args.logging_step == 0:
                loader.set_postfix(**{
                    'lr': optimizer.param_groups[0]['lr'],
                    'bce_score': '{:.4f}'.format(train_bce.avg),
                    'dice_score': '{:.4f}'.format(train_dice.avg),
                    'e_score': '{:.4f}'.format(train_e.avg),
                    'Loss': '{:.4f}'.format(train_loss.avg),
                })

        # 每两轮上传验证集指标以及保存
        if (epoch + 1) % args.validation_interval == 0:
            val_index = validation(val_dataloader, model)

            # 存储最好的模型
            if val_index['miou'] > best_miou:
                best_miou = val_index["miou"]
                save_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "arch": args.model_name,
                    "state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "optimizer": optimizer.state_dict()},
                    save_path
                )

            # 存储每一轮模型
            save_path = os.path.join(checkpoint_dir, "%s_checkpoint_%04depoch.pt" % (name, epoch + 1))
            torch.save({
                "epoch": epoch,
                "arch": args.model_name,
                "state_dict": model.state_dict(),
                "best_miou": best_miou,
                "optimizer": optimizer.state_dict()},
                save_path
            )
            # 在val_index添加epoch信息，并上传到wandb云平台
            val_index['epoch'] = epoch + 1
            # 更新学习率参数
            print(val_index)
            print("Saved checkpoint to: %s" % save_path)


def validation(valid_loader, model):
    # 声明需要用到的评价矩阵
    metric = SegmentationMetric(2, device)
    BCELoss = torch.nn.BCELoss()
    val_loss = metrics.MetricTracker()

    # 切换模式
    model.eval()

    loader = tqdm(valid_loader, desc="validation")
    for idx, data in enumerate(loader):
        image = data["image"].to(device)
        mask = data["mask"].to(device)
        # 输出结果
        predict = model(image).sigmoid()
        # 计算损失
        bce_score = BCELoss(predict, mask.unsqueeze(dim=1).float())
        val_loss.update(bce_score.item(), predict.size(0))
        # 计算混淆矩阵
        metric.add_batch((predict > 0.5).squeeze(dim=1).to(torch.int), mask)
        if idx % config.logging_step == 0:
            loader.set_postfix(**{'Loss': '{:.4f}'.format(val_loss.avg)})

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
    # 添加val_loss
    index_list = np.append(index_list, val_loss.avg)
    dict_index = dict(zip(label_name, index_list))

    # 打开model的训练模式
    model.train()

    return dict_index


if __name__ == "__main__":
    # 获取命令行参数
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    # 数据集设置
    parser.add_argument('--dataset', type=str, default='Massachusetts_Buildings_Dataset',
                        choices=['Massachusetts_Buildings_Dataset',
                                 'WHU_Building_Dataset'])
    parser.add_argument('--batch_size', default=4, type=int)
    # model参数
    parser.add_argument('--model_name', type=str, default='SPSNet', choices=['SPSNet', 'UMobileNet'])
    parser.add_argument("--pretrain", default=r"",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument('--in_channel', default=3, type=int, help="the number of input data channel")
    parser.add_argument('--out_channel', default=1, type=int, help="the number of classes")
    parser.add_argument('--spc', type=int, default=1, help="the channel of super pixel segment plug")
    # optimizer参数
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['RMSprop', 'Adam'])
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.01)")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # scheduler参数
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    # 其他参数
    parser.add_argument("--epochs", default=60, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--logging_step", type=int, default=20)

    parser.add_argument("--name", default="default", type=str, help="Experiment name")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(name=args.name)
