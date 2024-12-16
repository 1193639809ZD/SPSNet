import torch
from torch import nn

from model import mobilenetv2
from utils.super_pixel_utils import poolfeat


class SPSNet(nn.Module):
    def __init__(self, in_channel, out_channel, spc=3):
        super(SPSNet, self).__init__()
        self.in_channel = in_channel
        self.num_class = out_channel
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]
        backbone = mobilenetv2.mobilenet_v2(
            pretrained=True, interverted_residual_setting=interverted_residual_setting
        )
        backbone.features[0] = mobilenetv2.conv_bn(3, 32, 1)
        self.pred_mask = nn.Sequential(
            nn.Conv2d(spc, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 9, kernel_size=(3, 3), padding=1, bias=True),
            nn.Softmax(dim=1),
        )
        self.inc = backbone.features[:2]

        self.compress_channel1 = nn.Sequential(
            nn.Conv2d(16, spc, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spc),
        )
        self.down1 = backbone.features[2:4]

        self.compress_channel2 = nn.Sequential(
            nn.Conv2d(24, spc, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spc),
        )
        self.down2 = backbone.features[4:7]

        self.compress_channel3 = nn.Sequential(
            nn.Conv2d(32, spc, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spc),
        )
        self.down3 = backbone.features[7:11]

        self.compress_channel4 = nn.Sequential(
            nn.Conv2d(64, spc, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spc),
        )
        self.down4 = backbone.features[11:-1]

        self.up1 = Up(320, 64, [[6, 64, 4, 1]])
        self.up2 = Up(64, 32, [[6, 32, 3, 1]])
        self.up3 = Up(32, 24, [[6, 24, 2, 1]])
        self.up4 = Up(24, 16, [[1, 16, 2, 1]])
        self.outc = OutConv(16, out_channel)

        del backbone

    def forward(self, x):
        x1 = self.inc(x)

        x1_compress = self.compress_channel1(x1)
        mask1 = self.pred_mask(x1_compress)
        x2 = poolfeat(x1, mask1, 2, 2)
        x2 = self.down1(x2)

        x2_compress = self.compress_channel2(x2)
        mask2 = self.pred_mask(x2_compress)
        x3 = poolfeat(x2, mask2, 2, 2)
        x3 = self.down2(x3)

        x3_compress = self.compress_channel3(x3)
        mask3 = self.pred_mask(x3_compress)
        x4 = poolfeat(x3, mask3, 2, 2)
        x4 = self.down3(x4)

        x4_compress = self.compress_channel4(x4)
        mask4 = self.pred_mask(x4_compress)
        x5 = poolfeat(x4, mask4, 2, 2)
        x5 = self.down4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits, mask1, mask2, mask3, mask4


class UMobileNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UMobileNet, self).__init__()
        self.in_channel = in_channel
        self.num_class = out_channel
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        backbone = mobilenetv2.mobilenet_v2(
            pretrained=True, interverted_residual_setting=interverted_residual_setting
        )
        backbone.features[0] = mobilenetv2.conv_bn(3, 32, 1)
        self.inc = backbone.features[:2]
        self.down1 = backbone.features[2:4]
        self.down2 = backbone.features[4:7]
        self.down3 = backbone.features[7:11]
        self.down4 = backbone.features[11:-1]

        self.up1 = Up(320, 64, [[6, 64, 4, 1]])
        self.up2 = Up(64, 32, [[6, 32, 3, 1]])
        self.up3 = Up(32, 24, [[6, 24, 2, 1]])
        self.up4 = Up(24, 16, [[1, 16, 2, 1]])
        self.outc = OutConv(16, out_channel)

        # 删除backbo
        del backbone

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


class Up(nn.Module):
    def __init__(self, in_channels1, in_channel2, interverted_residual_setting):
        super().__init__()
        block = mobilenetv2.InvertedResidual
        self.conv = []
        for t, c, n, s in interverted_residual_setting:
            out_channels = c
            for i in range(n):
                if i == 0:
                    self.conv.append(block(in_channels1 + in_channel2, out_channels, s, expand_ratio=t))
                else:
                    self.conv.append(block(input_channel, out_channels, 1, expand_ratio=t))
                input_channel = out_channels
        self.up = nn.ConvTranspose2d(in_channels1, in_channels1, kernel_size=(2, 2), stride=(2, 2))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
