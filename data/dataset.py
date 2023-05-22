import torch
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import numpy as np


class ColorizeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, ids_path, transforms=None):
        self.transforms = transforms
        self.files = []
        with open(ids_path, 'r') as f:
            for line in f:
                self.files.append(data_path + line.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = np.array(self.transforms(img))
        img = rgb2lab(img).astype(np.float32)
        L = img[:, :, [0]] / 50.0 - 1.0
        ab = img[:, :, [1, 2]] / 110.0

        return {
            'L': torch.tensor(L.transpose(2, 0, 1), dtype=torch.float32),
            'ab': torch.tensor(ab.transpose(2, 0, 1), dtype=torch.float32)
        }


def get_images(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    imgs = torch.cat((L, ab), dim=1).permute(0, 2, 3, 1).cpu().numpy()

    return np.stack([
        lab2rgb(img) for img in imgs
    ], axis=0)
