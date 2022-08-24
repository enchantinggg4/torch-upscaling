
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from skimage import io, transform, color
import torch

class UpsampleDataset(Dataset):
    def __init__(self, path, i_image_size, o_image_size):
        self.root_dir = path
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]
        self.i_image_size = i_image_size
        self.o_image_size = o_image_size

        print(f'Got {len(self.images)} images')
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        
        image = io.imread(img_name)

        if image.shape[2] > 3:
            image = color.rgba2rgb(image)

        i_image = transform.resize(image, (self.i_image_size, self.i_image_size))
        o_image = transform.resize(image, (self.o_image_size, self.o_image_size))
        
        return i_image, o_image