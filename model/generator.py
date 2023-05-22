from torch import nn, cat


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stage, dropout, submodule):
        super(UNetBlock, self).__init__()
        self.stage = stage

        if stage == 0:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif stage == 1:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                submodule,
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels*2, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Dropout(0.5) if dropout else nn.Identity()
            )
        elif stage == 2:
            self.model = nn.Sequential(
                nn.Conv2d(1, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                submodule,
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels*2, 2, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )

    def forward(self, x):
        if self.stage == 2:
            return self.model(x)
        else:
            return cat([x, self.model(x)], dim=1)


class UNet(nn.Module):
    def __init__(self, num_layers=8, num_filters=64):
        super(UNet, self).__init__()
        block = UNetBlock(
            in_channels=num_filters*8, out_channels=num_filters*8,
            stage=0, dropout=False, submodule=None
        )

        for _ in range(num_layers-5):
            block = UNetBlock(
                in_channels=num_filters*8, out_channels=num_filters*8,
                stage=1, dropout=True, submodule=block
            )

        out_channels = num_filters*8
        for _ in range(3):
            block = UNetBlock(
                in_channels=out_channels//2, out_channels=out_channels,
                stage=1, dropout=False, submodule=block
            )
            out_channels //= 2

        self.model = UNetBlock(
            in_channels=1, out_channels=out_channels,
            stage=2, dropout=False, submodule=block
        )

    def forward(self, x):
        return self.model(x)
