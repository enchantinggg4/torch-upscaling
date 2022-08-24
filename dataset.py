
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from skimage import io, transform, color
import torch
from tqdm import tqdm
class UpsampleDataset(Dataset):
    def __init__(self, path, i_image_size, o_image_size):
        self.root_dir = path
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]
        self.i_image_size = i_image_size
        self.o_image_size = o_image_size

        

        self.x = []
        self.y = []

        print(f'Got {len(self.images)} images')


    def gpu_precache(self, device):
        for idx, img_name in tqdm(enumerate(self.images)):

            if idx > 2000:
                break
        
            image = io.imread(img_name)

            if image.shape[2] > 3:
                image = color.rgba2rgb(image)

            i_image = transform.resize(image, (self.i_image_size, self.i_image_size))
            o_image = transform.resize(image, (self.o_image_size, self.o_image_size))


            self.x.append(torch.tensor(i_image))
            self.y.append(torch.tensor(o_image))

        self.x = torch.stack(self.x).to(device)
        self.y = torch.stack(self.y).to(device)
        print('Device precache complete')
        
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]