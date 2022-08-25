from __future__ import print_function
from pathlib import Path

import torchvision.transforms as T
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

from model2 import Model2

workers = 0
nc = 3
nz = 25


lr = 1e-4
weight_decay = 1e-6

ngpu = 1

from dotenv import load_dotenv

NO_WANDB = False




def train(i_image_size, o_image_size, epochs, dataroot, batch_size, checkpoints, inplace_dataset):
    global NO_WANDB
    if 'NO_WANDB' in os.environ:
        NO_WANDB = True
    else:
        wandb.init(project="upscaling")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using device {device}')

    model = Model2(res_len=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    
    criterion = nn.L1Loss()

    print(f'{sum(p.numel() for p in model.parameters())} parameters')



    dataset = UpsampleDataset(dataroot, i_image_size, o_image_size, is_inplace=inplace_dataset)
    
    if inplace_dataset:
        pass
    else:
        dataset.gpu_precache(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=workers)

    model.train()
    for epoch in tqdm(range(0, epochs)):
        losses = np.array([])
        for i, data in enumerate(dataloader, 0):
            
            low_img = data[0].to(device, dtype=torch.float)
            high_img = data[1].to(device, dtype=torch.float)

            optimizer.zero_grad()

            out = model(low_img)
            loss = criterion(out, high_img)
            loss.backward()
            optimizer.step()


            losses = np.append(losses, loss.item())
            if not NO_WANDB:
                # NO_WANDB=true
                


                wandb.log({ 'loss': loss.item() })

                if i % 1 == 0:
                    slides = torch.cat((
                        T.Resize((o_image_size, o_image_size))(low_img[0:8]),
                        high_img[0:8],
                        out[0:8]))
                    samples = wandb.Image(slides, caption="Upscaled")
                    wandb.log({ 'samples': samples})
        print(f'Epoch {epoch}, Mean Loss: {np.mean(losses)}')

        if checkpoints:
            torch.save(model.state_dict(), f'./checkpoints/epoch_{epoch}_{np.mean(losses)}.pth')


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-nocp', action='store_true', dest='no_checkpoint')
    parser.add_argument('-inplace', action='store_true', dest='inplace_dataset')
    parser.add_argument('-data', action='store', dest='path')
    parser.add_argument('-batch', action='store', dest='batch_size', type=int)
    parser.add_argument('-epochs', action='store', dest='epochs', type=int)
    parser.add_argument('-i_size', action='store', dest='i_size', type=int)
    parser.add_argument('-o_size', action='store', dest='o_size', type=int)

    args = parser.parse_args()

    print(f'Using dataset {args.path}')
    print(f'Using batch size {args.batch_size}')
    print(f'Saving checkpoints: {not args.no_checkpoint}')
    print(f'Inplace dataset: {args.inplace_dataset}')

    print(f'Upscaling from {args.i_size}x{args.i_size} to {args.o_size}x{args.o_size}')

    print(f'Training for {args.epochs} epochs')
    Path('./checkpoints').mkdir(exist_ok=True)

    train(args.i_size, args.o_size, args.epochs, args.path, args.batch_size, not args.no_checkpoint, args.inplace_dataset)