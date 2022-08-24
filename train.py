from __future__ import print_function

import torch
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
import torchvision.utils as vutils
import numpy as np
from dataset import UpsampleDataset
from model import Model
from tqdm import tqdm
from skimage import io, transform
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

batch_size = 32
workers = 2
nc = 3
nz = 25


ngpu = 1


def train(i_image_size, o_image_size, dataroot):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using device {device}')
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.002)
    criterion = nn.L1Loss()


    dataset = UpsampleDataset(dataroot, i_image_size, o_image_size)
    dataset.gpu_precache(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    model.train()
    for epoch in tqdm(range(0, 10)):
        for i, data in enumerate(dataloader, 0):
            try:
            
                low_img = data[0].to(device, dtype=torch.float).permute(0, 3, 1, 2)
                high_img = data[1].to(device, dtype=torch.float).permute(0, 3, 1, 2)

                optimizer.zero_grad()

                out = model(low_img)
                loss = criterion(out, high_img)
                loss.backward()
                optimizer.step()


                if i != 0 and i % 25 == 0:
                    print(f'Iteration {i}, Loss: {loss.item()}')
                    
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-p', action='store', dest='path')

    args = parser.parse_args()

    train(64, 100, args.path)