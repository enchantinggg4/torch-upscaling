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
import torchvision.transforms as T
from PIL import Image



if __name__ == "__main__":
    Path('./generated').mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('-m', action='store', dest='model')
    parser.add_argument('-i', action='store', dest='input')

    args = parser.parse_args()

    print(f'Using version {args.model}')
    print(f'Upscaling file {args.input}')


    model = Model()
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    img = T.ToTensor()(Image.open(args.input).convert('RGB'))

    img = torch.unsqueeze(img, 0)
    T.ToPILImage()(img[0]).save('./generated/in.jpg')
    
    out = model(img)

    T.ToPILImage()(out[0]).save(f'./generated/out.jpg')
    