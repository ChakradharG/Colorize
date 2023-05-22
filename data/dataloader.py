from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import ColorizeDataset


def load_data(config):
    train_transforms = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip()
    ])
    train_set = ColorizeDataset(
        config.paths.root,
        config.paths.ids + 'train.txt',
        train_transforms
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Needed because real/fake labels are static (o/w shape mismatch in last batch)
    )

    test_transforms = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size))
    ])
    test_set = ColorizeDataset(
        config.paths.root,
        config.paths.ids + 'test.txt',
        test_transforms
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print('Batch size: ', config.batch_size)

    print('Train dataset samples: ', len(train_set))
    print('Test dataset samples: ', len(test_set))

    print('Train dataset batches: ', len(train_loader))
    print('Test dataset batches: ', len(test_loader))

    print()

    return train_loader, test_loader
