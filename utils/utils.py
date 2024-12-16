import torch
from torch.nn import functional as F
import cv2
import numpy as np

def feature_vis(feature, save_path):  
    # feaats形状: [1,c,h,w] 类型：Tensor，batch_size=1
    channel_mean = torch.mean(feature, dim=1, keepdim=True)  
    channel_mean = F.interpolate(channel_mean, size=(256, 256), mode='bilinear', align_corners=False)
    channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
    channel_mean = (((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(np.uint8)
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    cv2.imwrite(str(save_path), channel_mean)