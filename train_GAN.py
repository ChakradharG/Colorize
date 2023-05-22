import torch
import yaml
from easydict import EasyDict
from datetime import datetime
from os import makedirs
from shutil import copy
from data.dataloader import load_data
from model.network import create_model, setup_training
from model.utils import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


config = EasyDict(yaml.load(
    open('./config.yaml', 'r'), Loader=yaml.FullLoader
))
config.device = device
config.stage = 'train'

model = create_model(config)
if config.pretrained_gen:
    model[0].load_state_dict(torch.load(f'./checkpoints/{config.run_name}/pretrained_gen.pt'))

train_loader, test_loader = load_data(config)
criterion, optimizer, scheduler = setup_training(config, model)

if not config.debug:
    save_dir = f'./checkpoints/{config.run_name}'
    config.save_dir = save_dir

    makedirs(f'./viz/{config.run_name}', exist_ok=True)
    makedirs(save_dir, exist_ok=True)
    copy('./config.yaml', f'{save_dir}/config.yaml')
    copy('./model/network.py', f'{save_dir}/network.py')
    copy('./model/generator.py', f'{save_dir}/generator.py')
    copy('./model/discriminator.py', f'{save_dir}/discriminator.py')

train_model(
    config, model, train_loader, test_loader,
    criterion, optimizer, scheduler
)
