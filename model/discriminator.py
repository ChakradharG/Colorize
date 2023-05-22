from torch import nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, stage):
        super(DiscriminatorBlock, self).__init__()
        modules = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=(stage!=1))])

        if stage == 1:
            modules.append(nn.BatchNorm2d(out_channels))
        if stage != 2:
            modules.append(nn.LeakyReLU(0.2, True))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, num_layers=3, num_filters=64):
        super(PatchDiscriminator, self).__init__()
        modules = nn.ModuleList([DiscriminatorBlock(3, num_filters, 2, 0)])

        for _ in range(num_layers-1):
            modules.append(DiscriminatorBlock(num_filters, num_filters*2, 2, 1))
            num_filters *= 2

        self.model = nn.Sequential(
            *modules,
            DiscriminatorBlock(num_filters, num_filters*2, 1, 1),
            DiscriminatorBlock(num_filters*2, 1, 1, 2)
        )

    def forward(self, x):
        return self.model(x)
