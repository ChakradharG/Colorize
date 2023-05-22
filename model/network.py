from torch import nn, optim
from model.generator import UNet
from model.discriminator import PatchDiscriminator


def create_model(config):
    if config.stage == 'train':
        return [
            UNet(num_layers=8, num_filters=64).to(config.device),
            PatchDiscriminator(num_layers=3, num_filters=64).to(config.device)
        ]
    else:
        return UNet(num_layers=8, num_filters=64).to(config.device)


def setup_training(config, model):
    if config.stage == 'pretrain':
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.sch.factor,
            patience=config.sch.patience
        )
    else:
        criterion = [nn.BCEWithLogitsLoss(), nn.L1Loss()]
        optimizer = [
            optim.Adam(model[0].parameters(), lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2)),
            optim.Adam(model[1].parameters(), lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2))
        ]
        scheduler = [
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer[0],
                factor=config.sch.factor,
                patience=config.sch.patience
            ),
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer[1],
                factor=config.sch.factor,
                patience=config.sch.patience
            )
        ]

    return criterion, optimizer, scheduler
