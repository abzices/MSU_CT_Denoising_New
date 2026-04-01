import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class VGG19_DWT_VAE(nn.Module):
    def __init__(self, latent_channels=8):
        super(VGG19_DWT_VAE, self).__init__()

        # 1. 采用带 BN 层的 vgg19_bn，极其抗梯度爆炸！
        vgg19_bn = models.vgg19_bn(weights="IMAGENET1K_V1").features

        # 2. 魔改第一层卷积，适应 4 通道输入
        old_conv = vgg19_bn[0]
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = old_conv.weight
            self.conv1.weight[:, 3:4, :, :] = torch.mean(old_conv.weight, dim=1, keepdim=True)
            self.conv1.bias = old_conv.bias

        # 3. 动态截断 VGG19_bn：找到第 3 个 MaxPool2d 截断
        # 这样无论 VGG 结构怎么变，都能准确把 256x256 降维到 32x32
        features = [self.conv1]
        pool_count = 0
        for layer in list(vgg19_bn.children())[1:]:
            features.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                pool_count += 1
                if pool_count == 3:  # 到了第3个池化层就停止
                    break
        self.encoder_features = nn.Sequential(*features)  # 输出[Batch, 256, 32, 32]

        # 4. 映射到 Latent 空间（这些层现在会在 Stage1 参与训练了！）
        self.quant_conv_mu = nn.Conv2d(256, latent_channels, 1)
        self.quant_conv_logvar = nn.Conv2d(256, latent_channels, 1)

        # 5. 自定义 Decoder
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        )

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder_features(x)
        mu = self.quant_conv_mu(x)
        logvar = self.quant_conv_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar