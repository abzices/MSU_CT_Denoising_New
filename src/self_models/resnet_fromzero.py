import torch
import torch.nn as nn
from typing import Tuple


class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet瓶颈块（ResNet50使用）"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    """ResNet编码器：将4通道256×256图像压缩到潜在空间"""

    def __init__(self, block, layers, latent_channels=8):
        super(ResNetEncoder, self).__init__()
        self.in_channels = 64
        self.latent_channels = latent_channels

        # 初始卷积层：4通道输入
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 潜在空间映射
        self.quant_conv_mu = nn.Conv2d(512 * block.expansion, latent_channels, 1)
        self.quant_conv_logvar = nn.Conv2d(512 * block.expansion, latent_channels, 1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)  # [B, 64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, 64, 64]

        # ResNet层
        x = self.layer1(x)  # [B, 256, 64, 64] (Bottleneck) 或 [B, 64, 64, 64] (BasicBlock)
        x = self.layer2(x)  # [B, 512, 32, 32] (Bottleneck) 或 [B, 128, 32, 32] (BasicBlock)
        x = self.layer3(x)  # [B, 1024, 16, 16] (Bottleneck) 或 [B, 256, 16, 16] (BasicBlock)
        x = self.layer4(x)  # [B, 2048, 8, 8] (Bottleneck) 或 [B, 512, 8, 8] (BasicBlock)

        # 映射到潜在空间
        mu = self.quant_conv_mu(x)
        logvar = self.quant_conv_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar


class ResNetDecoder(nn.Module):
    """ResNet解码器：从潜在空间重建4通道256×256图像"""

    def __init__(self, block, layers, latent_channels=8):
        super(ResNetDecoder, self).__init__()
        self.in_channels = 512 * block.expansion
        self.latent_channels = latent_channels
        self.block_expansion = block.expansion

        # 从潜在空间映射回特征空间
        self.post_quant_conv = nn.Conv2d(latent_channels, 512 * block.expansion, 1)

        # 反向ResNet层（使用转置卷积上采样）
        self.deconv4 = self._make_deconv_layer(block, 512, layers[3], stride=2)
        self.deconv3 = self._make_deconv_layer(block, 256, layers[2], stride=2)
        self.deconv2 = self._make_deconv_layer(block, 128, layers[1], stride=2)
        self.deconv1 = self._make_deconv_layer(block, 64, layers[0], stride=2)

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(64 * block.expansion, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        )

    def _make_deconv_layer(self, block, out_channels, blocks, stride=1):
        """创建解码层（使用转置卷积）"""
        layers = []
        
        # 上采样卷积
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, out_channels * block.expansion,
                                 kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
                nn.ReLU(inplace=True)
            )
        )
        
        self.in_channels = out_channels * block.expansion
        
        # 添加残差块
        for _ in range(blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, z):
        # 从潜在空间映射
        x = self.post_quant_conv(z)  # [B, 2048, 8, 8]

        # 反向ResNet层
        x = self.deconv4(x)  # [B, 1024, 16, 16]
        x = self.deconv3(x)  # [B, 512, 32, 32]
        x = self.deconv2(x)  # [B, 256, 64, 64]
        x = self.deconv1(x)  # [B, 64, 128, 128]

        # 最终上采样到256×256
        x = self.final_conv(x)  # [B, 4, 256, 256]

        return x


class ResNet50Autoencoder(nn.Module):
    """
    基于ResNet50的编码器-解码器结构
    
    输入: 4通道256×256图像（由1×512×512的CT图像经小波变换得到）
    潜在空间: 8×8×8（可配置）
    输出: 4通道256×256重建图像
    """

    def __init__(self, latent_channels=8):
        super(ResNet50Autoencoder, self).__init__()
        
        # ResNet50使用Bottleneck块，层数配置为[3, 4, 6, 3]
        self.encoder = ResNetEncoder(Bottleneck, [3, 4, 6, 3], latent_channels)
        self.decoder = ResNetDecoder(Bottleneck, [3, 4, 6, 3], latent_channels)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码：输入→隐空间均值+对数方差"""
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """重参数化：从正态分布采样，保证梯度可回传"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 推理阶段直接用均值，避免随机性
            return mu

    def decode(self, z):
        """解码：隐空间→重建输出"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播：输入→重建输出+均值+对数方差"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ResNet18Autoencoder(nn.Module):
    """
    基于ResNet18的编码器-解码器结构（轻量级版本）
    
    输入: 4通道256×256图像
    潜在空间: 8×8×8（可配置）
    输出: 4通道256×256重建图像
    """

    def __init__(self, latent_channels=8):
        super(ResNet18Autoencoder, self).__init__()
        
        # ResNet18使用BasicBlock块，层数配置为[2, 2, 2, 2]
        self.encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2], latent_channels)
        self.decoder = ResNetDecoder(BasicBlock, [2, 2, 2, 2], latent_channels)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码：输入→隐空间均值+对数方差"""
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """重参数化：从正态分布采样，保证梯度可回传"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        """解码：隐空间→重建输出"""
        return self.decoder(z)

    def forward(self, x):
        """前向传播：输入→重建输出+均值+对数方差"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# 测试网络结构
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("测试 ResNet50 Autoencoder")
    print("=" * 60)
    
    # 测试ResNet50版本
    model50 = ResNet50Autoencoder(latent_channels=8).to(device)
    
    # 生成测试输入（batch=2, 4通道, 256x256）
    dummy_input = torch.randn(2, 4, 256, 256).to(device)
    
    # 前向传播
    model50.train()
    recon, mu, logvar = model50(dummy_input)
    
    # 打印结果
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"潜在空间均值尺寸: {mu.shape}")  # 预期: [2, 8, 8, 8]
    print(f"重建输出尺寸: {recon.shape}")  # 预期: [2, 4, 256, 256]
    
    # 计算参数量
    total_params = sum(p.numel() for p in model50.parameters())
    trainable_params = sum(p.numel() for p in model50.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("测试 ResNet18 Autoencoder（轻量级版本）")
    print("=" * 60)
    
    # 测试ResNet18版本
    model18 = ResNet18Autoencoder(latent_channels=8).to(device)
    
    # 前向传播
    model18.train()
    recon18, mu18, logvar18 = model18(dummy_input)
    
    # 打印结果
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"潜在空间均值尺寸: {mu18.shape}")  # 预期: [2, 8, 8, 8]
    print(f"重建输出尺寸: {recon18.shape}")  # 预期: [2, 4, 256, 256]
    
    # 计算参数量
    total_params18 = sum(p.numel() for p in model18.parameters())
    trainable_params18 = sum(p.numel() for p in model18.parameters() if p.requires_grad)
    print(f"总参数量: {total_params18:,}")
    print(f"可训练参数量: {trainable_params18:,}")
    
    print("\n" + "=" * 60)
    print("推理模式测试")
    print("=" * 60)
    
    # 验证推理模式（无随机性）
    model50.eval()
    with torch.no_grad():
        recon_eval1, mu_eval1, logvar_eval1 = model50(dummy_input)
        recon_eval2, mu_eval2, logvar_eval2 = model50(dummy_input)
    
    print(f"推理模式下两次重建结果是否相同: {torch.allclose(recon_eval1, recon_eval2)}")
    print(f"推理模式下两次潜在空间均值是否相同: {torch.allclose(mu_eval1, mu_eval2)}")
    
    print("\n测试完成！")
