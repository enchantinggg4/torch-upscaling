from __future__ import print_function
from pathlib import Path

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
import wandb

workers = 0
nc = 3
nz = 25


ngpu = 1

from dotenv import load_dotenv

NO_WANDB = False




def train(i_image_size, o_image_size, dataroot, batch_size):
    global NO_WANDB
    if 'NO_WANDB' in os.environ:
        NO_WANDB = True
    else:
        wandb.init(project="upscaling")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using device {device}')
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.002)
    # criterion = nn.L1Loss()
    criterion = nn.L1Loss()

    print(f'{sum(p.numel() for p in model.parameters())} parameters')



    dataset = UpsampleDataset(dataroot, i_image_size, o_image_size)
    dataset.gpu_precache(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    model.train()
    for epoch in tqdm(range(0, 10)):
        losses = np.array([])
        for i, data in enumerate(dataloader, 0):
            
            low_img = data[0].to(device, dtype=torch.float)#.permute(0, 3, 1, 2)
            high_img = data[1].to(device, dtype=torch.float)#.permute(0, 3, 1, 2)

            optimizer.zero_grad()

            out = model(low_img)
            loss = criterion(out, high_img)
            loss.backward()
            optimizer.step()


            losses = np.append(losses, loss.item())
            if not NO_WANDB:
                # NO_WANDB=true
                samples = wandb.Image(torch.cat((high_img[0:8], out[0:8])), caption="Upscaled")


                wandb.log({ 'loss': loss.item(), 'samples': samples })
        print(f'Epoch {epoch}, Mean Loss: {np.mean(losses)}')

        torch.save(model.state_dict(), f'./checkpoints/epoch_{epoch}_{np.mean(losses)}.pth')


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-p', action='store', dest='path')
    parser.add_argument('-b', action='store', dest='batch_size', type=int)

    args = parser.parse_args()

    print(f'Using dataset {args.path}')
    print(f'Using batch size {args.batch_size}')
    Path('./checkpoints').mkdir(exist_ok=True)
    train(64, 100, args.path, args.batch_size)