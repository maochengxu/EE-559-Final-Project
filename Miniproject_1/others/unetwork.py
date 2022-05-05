import torch
from torch import nn
from torch import functional as F


class UNetwork(nn.Module):
    """The U Network in Appendix A.1. Table 2
    """

    def __init__(self, in_channels=3, out_channels=3) -> None:
        super().__init__()
        # Encoder
        self.enc_conv0 = nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding='same')

        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding='same')
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding='same')
        self.pool_2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding='same')
        self.pool_3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding='same')
        self.pool_4 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding='same')
        self.pool_5 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv6 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding='same')

        # Decoder
        self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv5a = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding='same')
        self.dec_conv5b = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding='same')

        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv4a = nn.Conv2d(144, 96, kernel_size=3, stride=1, padding='same')
        self.dec_conv4b = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding='same')

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv3a = nn.Conv2d(144, 96, kernel_size=3, stride=1, padding='same')
        self.dec_conv3b = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding='same')

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, stride=1, padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding='same')

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv1a = nn.Conv2d(96+in_channels, 64, kernel_size=3, stride=1, padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same')
        self.dec_conv1c = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding='same')
    
    def forward(self, x) -> torch.Tensor:
        y = x
        y = F.leaky_relu(self.enc_conv0(y), 0.1)

        y = F.leaky_relu(self.enc_conv1(y), 0.1)
        y = self.pool_1(y)
        y_pool_1 = y

        y = F.leaky_relu(self.enc_conv2(y), 0.1)
        y = self.pool_2(y)
        y_pool_2 = y

        y = F.leaky_relu(self.enc_conv3(y), 0.1)
        y = self.pool_3(y)
        y_pool_3 = y

        y = F.leaky_relu(self.enc_conv4(y), 0.1)
        y = self.pool_4(y)
        y_pool_4 = y

        y = F.leaky_relu(self.enc_conv5(y), 0.1)
        y = self.pool_5(y)
        y = F.leaky_relu(self.enc_conv6(y), 0.1)

        y = self.upsample5(y)
        y = torch.cat((y, y_pool_4), dim=1)
        y = F.leaky_relu(self.dec_conv5a(y), 0.1)
        y = F.leaky_relu(self.dec_conv5b(y), 0.1)

        y = self.upsample4(y)
        y = torch.cat((y, y_pool_3), dim=1)
        y = F.leaky_relu(self.dec_conv4a(y), 0.1)
        y = F.leaky_relu(self.dec_conv4b(y), 0.1)

        y = self.upsample3(y)
        y = torch.cat((y, y_pool_2), dim=1)
        y = F.leaky_relu(self.dec_conv3a(y), 0.1)
        y = F.leaky_relu(self.dec_conv3b(y), 0.1)

        y = self.upsample2(y)
        y = torch.cat((y, y_pool_1), dim=1)
        y = F.leaky_relu(self.dec_conv2a(y), 0.1)
        y = F.leaky_relu(self.dec_conv2b(y), 0.1)

        y = self.upsample1(y)
        y = torch.cat((y, x), dim=1)
        y = F.leaky_relu(self.dec_conv1a(y), 0.1)
        y = F.leaky_relu(self.dec_conv1b(y), 0.1)
        y = self.dec_conv1c(y)

        return y
