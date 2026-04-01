import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

class ResNet50_DWT_VAE(nn.Module):
    def __init__(self, latent_channels=8):
        super(ResNet50_DWT_VAE, self).__init__()

        # 1.搭建encoder
        resnet = models.resnet50(weights="IMAGENET1K_V2")

        # 改造1：修改第一层卷积，将输入通道变为4（原为3）
        # 保留前3通道的预训练权重，第4通道权重求平均
        old_conv = resnet.conv1
        # 必须和官方原版 ResNet50 保持一模一样，才能完美继承权重
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = old_conv.weight
            self.conv1.weight[:, 3:4, :, :] = torch.mean(old_conv.weight, dim=1, keepdim=True)

        # 改造2：截断分类层，且不需要过深的下采样
        # 输入 256x256，经过 conv1->bn->relu->maxpool->layer1->layer2 后，尺寸变为 32x32，通道数为 512。
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1 # 输出: 256 x 64 x 64
        self.layer2 = resnet.layer2 # 输出: 512 x 32 x 32

        # 改造3：强制构建 VAE 的 Latent Space (均值和对数方差)
        # 将 512 通道压缩到我们想要的 latent_channels (如 8)
        self.quant_conv_mu = nn.Conv2d(512, latent_channels, 1)
        self.quant_conv_logvar = nn.Conv2d(512, latent_channels, 1)

        # 2.搭建decoder
        # 将latent_channels映射回512
        self.post_quant_conv = nn.Conv2d(latent_channels, 512, 1)

        # 逐层上采样
        self.decoder = nn.Sequential(
            # 32x32 -> 64x64
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),

            # 256*256 -> 恢复4个DWT通道
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
        )

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码：输入→隐空间均值+对数方差"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        mu = self.quant_conv_mu(x)
        logvar = self.quant_conv_logvar(x)

        # 限制logvar范围，防止数值爆炸（±10是经验值）
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化：从正态分布采样，保证梯度可回传"""
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            # 推理阶段直接用均值，避免随机性
            return mu

    def decode(self, z):
        """解码：隐空间→重建输出"""
        x = self.post_quant_conv(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """前向传播：输入→重建输出+均值+对数方差"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# 测试网络结构和损失计算
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = ResNet50_DWT_VAE(
        latent_channels=8
    ).to(device)

    # 生成测试输入（batch=2, 4通道, 256x256）
    dummy_input = torch.randn(2, 4, 256, 256).to(device)

    # 前向传播
    model.train()  # 训练模式（影响重参数化的随机性）
    recon, mu, logvar = model(dummy_input)

    # 打印结果
    print("=" * 50)
    print("输入尺寸:", dummy_input.shape)
    print("Latent均值尺寸:", mu.shape)  # 预期: [2, 8, 32, 32]
    print("重建输出尺寸:", recon.shape)  # 预期: [2, 4, 256, 256]
    print("=" * 50)
    print("=" * 50)

    # 验证推理模式（无随机性）
    model.eval()
    with torch.no_grad():
        recon_eval, mu_eval, logvar_eval = model(dummy_input)
    print("推理模式下重建输出均值:", recon_eval.mean().item())
