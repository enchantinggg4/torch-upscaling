import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import numpy as np
import torch.nn.functional as F

# x2
def conv_t_block_2x(in_ch, out_ch, dropout = 0.2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 3),
        nn.Dropout(dropout),
        
    )
# -4
def conv_block(in_ch, out_ch, dropout = 0.2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 0),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        
        dropout = 0.2
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.some = nn.Sequential(
             nn.Conv2d(3, 16, 3, 1, 1),
             nn.BatchNorm2d(16),
             self.act_fn,

             nn.Conv2d(16, 32, 3, 1, 1),
             nn.BatchNorm2d(32),
             self.act_fn,

             nn.Conv2d(32, 64, 3, 1, 1),
             nn.BatchNorm2d(64),
             self.act_fn,

             nn.Conv2d(64, 128, 3, 1, 1),
             nn.BatchNorm2d(128),
             self.act_fn,

             nn.Conv2d(128, 128, 3, 1, 1),
             nn.BatchNorm2d(128),
             self.act_fn,

             nn.Conv2d(128, 256, 3, 1, 1),
             nn.BatchNorm2d(256),
             self.act_fn,

             nn.Conv2d(256, 3, 3, 1, 1),
             nn.Tanh()
            #  nn.BatchNorm2d(256),
            #  self.act_fn,

            #  # in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=
            #  nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=3, bias=False),
            #  nn.Tanh()
        )
        
    def forward(self, x):
        return self.some(x)


print(Model3()(torch.randn(1, 3, 128, 128)).shape)