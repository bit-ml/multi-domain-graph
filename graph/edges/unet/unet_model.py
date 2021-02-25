""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

# class UNetLarge(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         # 17 mil params
#         super(UNetLarge, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits


class UNetMedium(nn.Module):
    def __init__(self, n_channels, n_classes, from_exp, to_exp):
        # 4 mil params
        super(UNetMedium, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.to_exp = to_exp

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, inp):
        x, post_proc_specific_fcn = inp
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
        return self.to_exp.post_process_ops(logits, post_proc_specific_fcn)


class UNetGood(nn.Module):
    def __init__(self, n_channels, n_classes, from_exp, to_exp):
        # 1 mil params
        super(UNetGood, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        self.to_exp = to_exp

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, inp):
        x, post_proc_specific_fcn = inp
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
        return self.to_exp.post_process_ops(logits, post_proc_specific_fcn)


# class UNetSmall(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         # 200k params
#         super(UNetSmall, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 32)
#         self.down1 = Down(32, 64)
#         factor = 2 if bilinear else 1
#         self.down2 = Down(64, 128 // factor)
#         self.up1 = Up(128, 64 // factor, bilinear)
#         self.up2 = Up(64, 32, bilinear)
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
#         logits = self.outc(x)
#         return logits
