
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import os
from PIL import Image
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

        i_transform = T.Resize((self.i_image_size, self.i_image_size))
        o_transform = T.Resize((self.o_image_size, self.o_image_size))
        to_tensor = T.ToTensor()
        for idx, img_name in tqdm(enumerate(self.images)):

            if idx > 2000:
                break
        
            image = Image.open(img_name).convert('RGB')

            # if image.shape[2] > 3:
            #     image = color.rgba2rgb(image)

            i_image = to_tensor(i_transform(image))
            o_image = to_tensor(o_transform(image))

            self.x.append(i_image)
            self.y.append(o_image)

        self.x = torch.stack(self.x).to(device)
        self.y = torch.stack(self.y).to(device)
        print('Device precache complete')
        
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]