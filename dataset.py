
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import os
from PIL import Image
from skimage import io, transform, color
import torch
from tqdm import tqdm
from pathlib import Path
from torchvision.utils import save_image

class UpsampleDataset(Dataset):
    def __init__(self, path, i_image_size, o_image_size):
        self.root_dir = path
        
        self.i_image_size = i_image_size
        self.o_image_size = o_image_size

        self.x = []
        self.y = []


    def transform_dataset(self):
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]
        i_transform = T.Resize((self.i_image_size, self.i_image_size))
        o_transform = T.Resize((self.o_image_size, self.o_image_size))
        
        
        for idx, img_name in tqdm(enumerate(self.images)):
            image = Image.open(img_name).convert('RGB')
            i_image = i_transform(image)
            o_image = o_transform(image)

            i_image.save(f'dataset/x/{idx}.jpg')
            o_image.save(f'dataset/y/{idx}.jpg')



    def gpu_precache(self, device):

        x_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(self.root_dir, 'x')) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]
        y_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(self.root_dir, 'y')) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]

        self.len = min(len(x_images), len(y_images))
        print(f'Got {self.len} images')
        i_transform = T.Resize((self.i_image_size, self.i_image_size))
        o_transform = T.Resize((self.o_image_size, self.o_image_size))
        to_tensor = T.ToTensor()

        for idx in tqdm(range(self.len)):
            i_image = Image.open(os.path.join(self.root_dir, 'x', f'{idx}.jpg')).convert('RGB')
            o_image = Image.open(os.path.join(self.root_dir, 'x', f'{idx}.jpg')).convert('RGB')


            i_image = i_transform(to_tensor(i_image))
            o_image = o_transform(to_tensor(o_image))

            self.x.append(i_image)
            self.y.append(o_image)

        self.x = torch.stack(self.x).to(device)
        self.y = torch.stack(self.y).to(device)
        print('Device precache complete')
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-p', action='store', dest='path')

    args = parser.parse_args()


    # transform all images to resized
    Path('./dataset/x').mkdir(parents=True, exist_ok=True)
    Path('./dataset/y').mkdir(parents=True, exist_ok=True)
    
    UpsampleDataset(args.path, 64, 100).transform_dataset()
