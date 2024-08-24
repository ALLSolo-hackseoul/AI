import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        self.conv1 = nn.Sequential([
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        ])
        self.conv2 = nn.Sequential([
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        self.trans_conv1 = nn.Sequential([
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=4),
            nn.GELU()
        ])
        self.trans_conv2 = nn.Sequential([
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=4),
            nn.Sigmoid()
        ])

    def forward(self, x):
        x = self.trans_conv1(x)
        return self.trans_conv2(x)
