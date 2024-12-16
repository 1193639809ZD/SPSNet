# 数据处理库
import numpy as np
import torch

# 模型代码
from model import spsnet

# 数据集代码
from datasets import WHUBuilding
from torch.utils.data import DataLoader

# 信息输出代码
from tqdm import tqdm

# 图片处理代码
from PIL import Image
import imgviz

# 其他
from pathlib import Path
from torch.nn import functional as F


def label2one_hot_torch(labels, C=14):
    # w.r.t http://jacobkimmel.github.io/pytorch_onehot/
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    """
    b, _, h, w = labels.shape
    one_hot = torch.zeros(b, C, h, w, dtype=torch.long).cuda()
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1)  # require long type

    return target.type(torch.float32)


def shift9pos(input, h_shift_unit=1, w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode="edge")
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top = input_pd[:, : -2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    bottom = input_pd[:, 2 * h_shift_unit :, w_shift_unit:-w_shift_unit]
    left = input_pd[:, h_shift_unit:-h_shift_unit, : -2 * w_shift_unit]
    right = input_pd[:, h_shift_unit:-h_shift_unit, 2 * w_shift_unit :]

    center = input_pd[:, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]

    bottom_right = input_pd[:, 2 * h_shift_unit :, 2 * w_shift_unit :]
    bottom_left = input_pd[:, 2 * h_shift_unit :, : -2 * w_shift_unit]
    top_right = input_pd[:, : -2 * h_shift_unit, 2 * w_shift_unit :]
    top_left = input_pd[:, : -2 * h_shift_unit, : -2 * w_shift_unit]

    shift_tensor = np.concatenate(
        [top_left, top, top_right, left, center, right, bottom_left, bottom, bottom_right], axis=0
    )
    return shift_tensor


def init_spixel_grid(b_train=True):
    if b_train:
        img_height, img_width = 512, 512
    else:
        img_height, img_width = 512, 512

    # get spixel id for the final assignment
    n_spixl_h = int(np.floor(img_height / 2))
    n_spixl_w = int(np.floor(img_width / 2))

    spixel_height = int(img_height / (1.0 * n_spixl_h))
    spixel_width = int(img_width / (1.0 * n_spixl_w))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(np.repeat(spix_idx_tensor_, spixel_height, axis=1), spixel_width, axis=2)

    torch_spix_idx_tensor = torch.from_numpy(np.tile(spix_idx_tensor, (2, 1, 1, 1))).type(torch.float).cuda()

    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    # pixel coord
    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing="ij"))

    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])

    all_XY_feat = torch.from_numpy(np.tile(coord_tensor, (2, 1, 1, 1)).astype(np.float32)).cuda()

    return torch_spix_idx_tensor, all_XY_feat


def build_LABXY_feat(label_in, XY_feat):

    img_lab = label_in.clone().type(torch.float)

    b, _, curr_img_height, curr_img_width = XY_feat.shape
    scale_img = F.interpolate(img_lab, size=(curr_img_height, curr_img_width), mode="nearest")
    LABXY_feat = torch.cat([scale_img, XY_feat], dim=1)

    return LABXY_feat


def poolfeat(input, prob, sp_h=2, sp_w=2):

    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w
    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    send_to_top_left = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, 2 * h_shift_unit :, 2 * w_shift_unit :
    ]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, 2 * h_shift_unit :, w_shift_unit:-w_shift_unit
    ]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode="constant", value=0)[:, :, 2 * h_shift_unit :, : -2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit :
    ]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit
    ]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, h_shift_unit:-h_shift_unit, : -2 * w_shift_unit
    ]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, : -2 * h_shift_unit, 2 * w_shift_unit :
    ]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, : -2 * h_shift_unit, w_shift_unit:-w_shift_unit
    ]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(
        feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)
    )  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode="constant", value=0)[
        :, :, : -2 * h_shift_unit, : -2 * w_shift_unit
    ]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat


def upfeat(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode="constant", value=0)

    gt_frm_top_left = F.interpolate(
        feat_pd[:, :, : -2 * h_shift, : -2 * w_shift], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

    top = F.interpolate(
        feat_pd[:, :, : -2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(
        feat_pd[:, :, : -2 * h_shift, 2 * w_shift :], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum += top_right * prob.narrow(1, 2, 1)

    left = F.interpolate(
        feat_pd[:, :, h_shift:-w_shift, : -2 * w_shift], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode="nearest")
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(
        feat_pd[:, :, h_shift:-w_shift, 2 * w_shift :], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(
        feat_pd[:, :, 2 * h_shift :, : -2 * w_shift], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(
        feat_pd[:, :, 2 * h_shift :, w_shift:-w_shift], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right = F.interpolate(
        feat_pd[:, :, 2 * h_shift :, 2 * w_shift :], size=(h * up_h, w * up_w), mode="nearest"
    )
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


def compute_semantic_pos_loss(prob_in, labxy_feat, pos_weight=0.003, kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:, -2:, :, :] - labxy_feat[:, -2:, :, :]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = -torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum = 0.005 * (loss_sem + loss_pos)
    loss_sem_sum = 0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum, loss_pos_sum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)


if __name__ == "__main__":
    # 加载模型
    model = spsnet.__dict__["SPSNet"](in_channel=3, out_channel=1, spc=3).cuda()
    # 加载预训练参数
    checkpoint = torch.load(r"checkpoints/test/test_checkpoint_0001epoch.pt")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    del checkpoint

    # 加载数据集
    dataset_val = WHUBuilding(root=Path("/mnt/d/dataset/WHU/"), mode="test")
    val_dataloader = DataLoader(dataset_val, batch_size=2, num_workers=2, shuffle=False)

    loader = tqdm(val_dataloader)
    spixelID, XY_feat_stack = init_spixel_grid()
    total_loss = AverageMeter()
    losses_sem = AverageMeter()
    losses_pos = AverageMeter()
    with torch.no_grad():
        for data in loader:
            image = data["image"].cuda()
            mask = data["mask"].cuda()

            # mask 增加一个维度
            mask = mask.unsqueeze(1)
            label_1hot = label2one_hot_torch(mask, C=2)
            LABXY_feat_tensor = build_LABXY_feat(label_1hot, XY_feat_stack)
            output = model(image)
            slic_loss, loss_sem, loss_pos = compute_semantic_pos_loss(
                output[1], LABXY_feat_tensor, pos_weight=0.003, kernel_size=2
            )

            total_loss.update(slic_loss.item(), image.size(0))
            losses_sem.update(loss_sem.item(), image.size(0))
            losses_pos.update(loss_pos.item(), image.size(0))

    print("total_loss:", total_loss)
    print("losses_sem:", losses_sem)
    print("losses_pos:", losses_pos)
