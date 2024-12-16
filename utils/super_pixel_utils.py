import numpy as np
import torch
import torch.nn.functional as F


def init_spixel_grid(img_width, img_height, downsize, batch_size):
    # get spixel id for the final assignment
    n_spixl_h = int(np.floor(img_height / downsize))
    n_spixl_w = int(np.floor(img_width / downsize))

    spixel_height = int(img_height / (1. * n_spixl_h))
    spixel_width = int(img_width / (1. * n_spixl_w))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, spixel_height, axis=1), spixel_width, axis=2)

    torch_spix_idx_tensor = torch.from_numpy(
        np.tile(spix_idx_tensor, (batch_size, 1, 1, 1))).type(torch.float).cuda()

    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    # pixel coord
    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))

    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])

    all_XY_feat = (torch.from_numpy(
        np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)).cuda())

    return torch_spix_idx_tensor, all_XY_feat


# feat manipulate
def poolfeat(input, prob, sp_h=2, sp_w=2):
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum = feat_sum + shift_feat[:, :-1, :, :]
        prob_sum = prob_sum + shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    # input_pad = F.pad(input, p2d, mode='constant', value=0)

    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).to(input.device)], dim=1)  # b* (n+1) *h*w

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    send_to_top_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit,
             w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat


def poolfeat_hard(input, prob_in, sp_h=2, sp_w=2):
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum = feat_sum + shift_feat[:, :-1, :, :]
        prob_sum = prob_sum + shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    prob = prob_in.clone()
    b, _, h, w = input.shape
    assig_max, prob_idx = torch.max(prob, dim=1, keepdim=True)
    prob = torch.where(prob == assig_max, torch.ones_like(prob).to(prob.device), torch.zeros_like(prob).to(prob.device))

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    # input_pad = F.pad(input, p2d, mode='constant', value=0)

    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).to(input.device)], dim=1)  # b* (n+1) *h*w

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    send_to_top_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit,
             w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat


def upfeat(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    # shift_in = shift9pos_torch(input).permute(0, 4, 1, 2, 3)  # b*9*c*H*W
    h_shift = up_h
    w_shift = up_w

    # upsample_feat = F.interpolate(input, size=(h * up_h, w * up_w), mode='nearest')
    upsample_feat = my_repeat(input, up_h, up_w)

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(upsample_feat, p2d, mode='constant', value=0)

    # param_pd = F.pad(input, p2d, mode='replicate')

    # assign val to ... (val to that location)
    # aaa = feat_pd[:, :, :-2 * h_shift, :-2 * w_shift]
    # bb = F.interpolate(aaa,scale_factor=(up_h, up_w),mode='nearest')
    gt_frm_top_left = feat_pd[:, :, :-2 * h_shift, :-2 * w_shift]
    feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

    top = feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift]
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = feat_pd[:, :, :-2 * h_shift, 2 * w_shift:]
    feat_sum += top_right * prob.narrow(1, 2, 1)

    left = feat_pd[:, :, h_shift:-h_shift, :-2 * w_shift]
    feat_sum += left * prob.narrow(1, 3, 1)

    center = upsample_feat
    feat_sum += center * prob.narrow(1, 4, 1)

    right = feat_pd[:, :, h_shift:-h_shift, 2 * w_shift:]
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = feat_pd[:, :, 2 * h_shift:, :-2 * w_shift]
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift]
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right = feat_pd[:, :, 2 * h_shift:, 2 * w_shift:]
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


def upfeat_hard(input, prob_in, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    # b, c, h, w = input.shape
    prob = prob_in.clone()
    # b, _, h, w = input.shape
    assig_max, prob_idx = torch.max(prob, dim=1, keepdim=True)
    prob = torch.where(prob == assig_max, torch.ones_like(prob).to(prob.device), torch.zeros_like(prob).to(prob.device))

    # shift_in = shift9pos_torch(input).permute(0, 4, 1, 2, 3)  # b*9*c*H*W
    h_shift = up_h
    w_shift = up_w

    # upsample_feat = F.interpolate(input, size=(h * up_h, w * up_w), mode='nearest')
    upsample_feat = my_repeat(input, up_h, up_w)

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(upsample_feat, p2d, mode='constant', value=0)

    # param_pd = F.pad(input, p2d, mode='replicate')

    # assign val to ... (val to that location)
    # aaa = feat_pd[:, :, :-2 * h_shift, :-2 * w_shift]
    # bb = F.interpolate(aaa,scale_factor=(up_h, up_w),mode='nearest')
    gt_frm_top_left = feat_pd[:, :, :-2 * h_shift, :-2 * w_shift]
    feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

    top = feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift]
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = feat_pd[:, :, :-2 * h_shift, 2 * w_shift:]
    feat_sum += top_right * prob.narrow(1, 2, 1)

    left = feat_pd[:, :, h_shift:-h_shift, :-2 * w_shift]
    feat_sum += left * prob.narrow(1, 3, 1)

    center = upsample_feat
    feat_sum += center * prob.narrow(1, 4, 1)

    right = feat_pd[:, :, h_shift:-h_shift, 2 * w_shift:]
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = feat_pd[:, :, 2 * h_shift:, :-2 * w_shift]
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift]
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right = feat_pd[:, :, 2 * h_shift:, 2 * w_shift:]
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


#   LABXY feature for spixel training
def rgb2Lab_torch(img_in, mean_values=None, std=None):
    # inpu img should be [0,1] float b*3*h*w

    img = (img_in.clone() * std.to(img_in.device) + mean_values.to(img_in.device)).clamp(0, 1)
    # else:
    #     img = (img_in.clone()).clamp(0, 1)
    # img = img_in.clone()

    mask = img > 0.04045
    img[mask] = torch.pow((img[mask] + 0.055) / 1.055, 2.4)
    img[~mask] /= 12.92

    xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                                 [0.212671, 0.715160, 0.072169],
                                 [0.019334, 0.119193, 0.950227]]).to(img_in.device)
    rgb = img.permute(0, 2, 3, 1)

    xyz_img = torch.matmul(rgb, xyz_from_rgb.transpose_(0, 1))

    xyz_ref_white = torch.tensor([0.95047, 1., 1.08883]).to(img_in.device)

    # scale by CIE XYZ tristimulus values of the reference white point
    lab = xyz_img / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = lab > 0.008856
    lab[mask] = torch.pow(lab[mask], 1. / 3.)
    lab[~mask] = 7.787 * lab[~mask] + 16. / 116.

    x, y, z = lab[..., 0:1], lab[..., 1:2], lab[..., 2:3]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return torch.cat([L, a, b], dim=-1).permute(0, 3, 1, 2)


def build_LABXY_feat(img_in, XY_feat):
    img_lab = img_in.type(torch.float)
    LABXY_feat = torch.cat([img_lab, XY_feat], dim=1)

    return LABXY_feat


# torch repeat
def my_repeat(x, h_unit, w_unit):
    tmp = x.clone()
    b, c, h, w = x.shape
    # for w direction
    tmp_w = tmp.view(b, c, -1, 1).repeat(1, 1, 1, w_unit).view(b, c, h, w * w_unit)
    # for h direction
    tmp_h = tmp_w.permute(0, 1, 3, 2).contiguous().view(b, c, -1, 1).repeat(1, 1, 1, h_unit).view(b, c, w * w_unit,
                                                                                                  h * h_unit)
    return tmp_h.permute(0, 1, 3, 2)


# init function

def shift9pos(input, h_shift_unit=1, w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top = input_pd[:, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    bottom = input_pd[:, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    left = input_pd[:, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    right = input_pd[:, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]

    center = input_pd[:, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]

    bottom_right = input_pd[:, 2 * h_shift_unit:, 2 * w_shift_unit:]
    bottom_left = input_pd[:, 2 * h_shift_unit:, :-2 * w_shift_unit]
    top_right = input_pd[:, :-2 * h_shift_unit, 2 * w_shift_unit:]
    top_left = input_pd[:, :-2 * h_shift_unit, :-2 * w_shift_unit]

    shift_tensor = np.concatenate([top_left, top, top_right,
                                   left, center, right,
                                   bottom_left, bottom, bottom_right], axis=0)
    return shift_tensor
