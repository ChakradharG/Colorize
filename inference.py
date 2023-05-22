import torch
import yaml
from easydict import EasyDict
from PIL import Image
from os import makedirs, listdir
from model.network import create_model
import numpy as np
from data.dataset import get_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

checkpoint = 'run1'
input_dir = './ip'
output_dir = './op'
makedirs(output_dir, exist_ok=True)


config = EasyDict(yaml.load(
    open('./config.yaml', 'r'), Loader=yaml.FullLoader
))
config.device = device
config.stage = 'inference'

model = create_model(config)
model.load_state_dict(torch.load(f'./checkpoints/{checkpoint}/model.pt', map_location=device)['gen_state_dict'])
model.eval()

def colorize():
    with torch.inference_mode():
        for img in listdir(input_dir):
            image = Image.open(f'{input_dir}/{img}').convert('L')
            image = image.resize((config.image_size, config.image_size))
            image = np.array(image, dtype=np.float32)
            L = torch.tensor(2 * (image / 255) - 1)
            L = L.unsqueeze(0).unsqueeze(0).to(device)
            image = get_images(L, model(L))[0]
            image = 255 * (image - image.min()) / (image.max() - image.min())
            Image.fromarray(image.astype(np.uint8)).save(f'{output_dir}/{img}')

colorize()
