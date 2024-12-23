import torch
import torch.nn as nn
from torchvision import models

class MobileNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetUNet, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True).features

        # Decoder layers
        self.decoder1 = self._decoder_block(1280, 512)
        self.decoder2 = self._decoder_block(512, 256)
        self.decoder3 = self._decoder_block(256, 128)
        self.decoder4 = self._decoder_block(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.backbone[:4](x)  # MobileNet downsampling layers
        x2 = self.backbone[4:](x1)  # MobileNet deep layers

        x = self.decoder1(x2)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.final_conv(x)
        return x

# Instantiate the model
model = MobileNetUNet(num_classes=1)  # for binary segmentation
