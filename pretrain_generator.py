import torch
import yaml
from tqdm import tqdm
from easydict import EasyDict
from os import makedirs
from data.dataloader import load_data
from model.network import create_model, setup_training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


config = EasyDict(yaml.load(
    open('./config.yaml', 'r'), Loader=yaml.FullLoader
))
config.device = device
config.stage = 'pretrain'

model = create_model(config)
train_loader, _ = load_data(config)
criterion, optimizer, scheduler = setup_training(config, model)

if not config.debug:
    save_dir = f'./checkpoints/{config.run_name}'
    makedirs(save_dir, exist_ok=True)

for epoch in range(config.epochs):
    bar = tqdm(train_loader, desc='Pretrain', leave=False, dynamic_ncols=True)
    total_loss = 0.0
    for i, batch in enumerate(bar):
        L = batch['L'].to(device)
        ab = batch['ab'].to(device)

        optimizer.zero_grad()
        ab_ = model(L)  # generated ab
        loss = criterion(ab_, ab)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f'{total_loss / (i + 1):.3f}')
        bar.update()
    bar.close()
        
    print(f'Epoch {epoch+1}/{config.epochs} - Pretrain Loss: {total_loss / len(train_loader):.3f}\n')

    scheduler.step(total_loss)

    if not config.debug:
        torch.save(model.state_dict(), f'{save_dir}/pretrained_gen.pt')
