import torch
import wandb
from tqdm import tqdm
from matplotlib import pyplot as plt
from data.dataset import get_images


def train(model, dataloader, criterion, optimizer, device, real_labels, fake_labels, lmbd):
    model[0].train()
    model[1].train()
    bar = tqdm(dataloader, desc='Train', leave=False, dynamic_ncols=True)
    total_loss = [0.0, 0.0]
    for i, batch in enumerate(bar):
        L = batch['L'].to(device)
        ab = batch['ab'].to(device)

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        gen_op = model[0](L)   # generated ab
        ab_ = gen_op.detach().clone().requires_grad_(True)  # breaking the gradient flow
        fake_preds = model[1](torch.cat((L, ab_), dim=1))

        # Generator
        gen_loss = criterion[0](fake_preds, real_labels) + lmbd * criterion[1](ab_, ab)
        gen_loss.backward(retain_graph=True)
        gen_op.backward(ab_.grad)
        optimizer[0].step()

        # Discriminator
        optimizer[1].zero_grad()
        real_preds = model[1](torch.cat((L, ab), dim=1))
        dis_loss = (criterion[0](fake_preds, fake_labels) + criterion[0](real_preds, real_labels)) / 2
        dis_loss.backward()
        optimizer[1].step()

        total_loss[0] += gen_loss.item()
        total_loss[1] += dis_loss.item()

        bar.set_postfix(
            gen_loss=f'{total_loss[0] / (i + 1):.3f}',
            dis_loss=f'{total_loss[1] / (i + 1):.3f}',
            gen_lr=f'{optimizer[0].param_groups[0]["lr"]:.3e}',
            dis_lr=f'{optimizer[1].param_groups[0]["lr"]:.3e}'
        )

        bar.update()
    bar.close()

    return total_loss[0] / len(dataloader), total_loss[1] / len(dataloader)


def test(model, dataloader, criterion, device, real_labels, fake_labels, lmbd):
    model[0].eval()
    model[1].eval()
    bar = tqdm(dataloader, desc='Test', leave=False, dynamic_ncols=True)
    total_loss = [0.0, 0.0]
    with torch.inference_mode():
        for i, batch in enumerate(bar):
            L = batch['L'].to(device)
            ab = batch['ab'].to(device)

            ab_ = model[0](L)
            fake_preds = model[1](torch.cat((L, ab_), dim=1))
            real_preds = model[1](torch.cat((L, ab), dim=1))

            gen_loss = criterion[0](fake_preds, real_labels) + lmbd * criterion[1](ab_, ab)
            dis_loss = (criterion[0](fake_preds, fake_labels) + criterion[0](real_preds, real_labels)) / 2

            total_loss[0] += gen_loss.item()
            total_loss[1] += dis_loss.item()

            bar.set_postfix(
                gen_loss=f'{total_loss[0] / (i + 1):.3f}',
                dis_loss=f'{total_loss[1] / (i + 1):.3f}'
            )

            bar.update()
        bar.close()

    return total_loss[0] / len(dataloader), total_loss[1] / len(dataloader)


def train_model(config, model, train_loader, test_loader, criterion, optimizer, scheduler):
    if not config.debug:
        wandb.init(project='Colorize', name=config.run_name)
    best_loss = [float('inf'), float('inf')]

    with torch.inference_mode():
        tmp = next(iter(train_loader))
        img = torch.cat((tmp['L'], tmp['ab']), dim=1).to(config.device)
        label_shape = model[1](img).shape
    fake_labels = torch.zeros(label_shape).to(config.device)
    real_labels = torch.ones(label_shape).to(config.device)

    for epoch in range(config.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, config.device, real_labels, fake_labels, config.optim.lmbd)
        print(f'Epoch {epoch+1}/{config.epochs} - Train - Generator Loss: {train_loss[0]:.3f}\tDiscriminator Loss: {train_loss[1]:.3f}')

        test_loss = test(model, test_loader, criterion, config.device, real_labels, fake_labels, config.optim.lmbd)
        print(f'Epoch {epoch+1}/{config.epochs} - Test - Generator Loss: {test_loss[0]:.3f}\tDiscriminator Loss: {test_loss[1]:.3f}\n')

        if epoch % 10 == 0:
            save_images(model[0], test_loader, config, epoch)

        if not config['debug']:
            wandb.log({
                'train_gen_loss': train_loss[0],
                'train_dis_loss': train_loss[1],
                'test_gen_loss': test_loss[0],
                'test_dis_loss': test_loss[1],
                'gen_lr': optimizer[0].param_groups[0]['lr'],
                'dis_lr': optimizer[1].param_groups[0]['lr']
            })

            if (test_loss[0] + test_loss[1]) <= (best_loss[0] + best_loss[1]):
                best_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'gen_state_dict': model[0].state_dict(),
                    'dis_state_dict': model[1].state_dict(),
                    'gen_optimizer_state_dict': optimizer[0].state_dict(),
                    'dis_optimizer_state_dict': optimizer[1].state_dict(),
                    'gen_scheduler_state_dict': scheduler[0].state_dict(),
                    'dis_scheduler_state_dict': scheduler[1].state_dict(),
                    'train_gen_loss': train_loss[0],
                    'train_dis_loss': train_loss[1],
                    'test_gen_loss': test_loss[0],
                    'test_dis_loss': test_loss[1],
                }, f'{config.save_dir}/model.pt')
                print('Model Saved\n')

        scheduler[0].step(test_loss[0])
        scheduler[1].step(test_loss[1])

    if not config.debug:
        wandb.finish()


def save_images(generator, dataloader, config, epoch):
    with torch.inference_mode():
        tmp = next(iter(dataloader))
        L = tmp['L'].to(config.device)
        ab = tmp['ab'].to(config.device)
        ab_ = generator(L)
        fake_imgs = get_images(L, ab_)
        real_imgs = get_images(L, ab)

    for i in range(config.batch_size):
        plt.subplot(3, config.batch_size, 1 + i)
        plt.imshow(L[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.axis('off')

    for i in range(config.batch_size):
        plt.subplot(3, config.batch_size, 5 + i)
        plt.imshow(fake_imgs[i])
        plt.axis('off')

    for i in range(config.batch_size):
        plt.subplot(3, config.batch_size, 9 + i)
        plt.imshow(real_imgs[i])
        plt.axis('off')

    plt.savefig(f'./viz/{config.run_name}/{epoch}.png')
