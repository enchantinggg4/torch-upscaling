import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import numpy as np
import torch.nn.functional as F

from util.modules import _Residual_Block


class Model2(nn.Module):
    def __init__(self, res_len):
        super(Model2, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = nn.Sequential(*[
            _Residual_Block() for x in range(res_len)
        ])

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)


        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)


    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))

        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)



        return out