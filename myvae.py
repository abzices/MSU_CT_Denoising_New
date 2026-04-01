import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
import pydicom


def ct_preprocess(ct_data):
    """
    修复版：支持 numpy数组 / torch张量 输入
    ct_data: 原始CT数据 (512,512) 或 (1,512,512)，DCM读取的HU值
    返回: 归一化后的 torch张量 (1,512,512)
    """
    # 🔥 核心修复：自动把 numpy 转换为 torch 张量
    if isinstance(ct_data, np.ndarray):
        ct_tensor = torch.from_numpy(ct_data).float()
    else:
        ct_tensor = ct_data.float()

    # 确保维度是 [1,512,512] (单通道)
    if ct_tensor.dim() == 2:
        ct_tensor = ct_tensor.unsqueeze(0)  # (512,512) → (1,512,512)

    # CT 窗宽窗位裁剪（软组织窗，可根据你的CT部位修改）
    WL, WW = 50, 350
    ct_tensor = torch.clip(ct_tensor, WL - WW // 2, WL + WW // 2)

    # 归一化到 [-1, 1] (LDM/VAE标准)
    ct_tensor = (ct_tensor - ct_tensor.min()) / (ct_tensor.max() - ct_tensor.min()) * 2 - 1
    return ct_tensor

def read_ct_dcm(dcm_path):
    """
    读取单张DCM格式CT文件 → 返回HU值numpy数组 (512,512)
    dcm_path: 你的dcm文件路径
    """
    # 读取DCM文件
    dcm = pydicom.dcmread(dcm_path)
    # 🔥 关键：CT像素值 → 标准HU值（医学CT必须这一步）
    pixel_array = dcm.pixel_array
    hu_values = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    return hu_values

class ResNetEncoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()

        # 加载与训练Resnet50,除去分类头
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 将原生第一层Conv2d（3,64，...)改为Conv2d（1,64，...）
        self.resnet.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)

        # 去掉ResNet的全局池化+全连接层（只保留特征提取主干）
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])

        # ============== 关键2：LDM-VAE 潜在空间映射 ==============
        # 输出潜在空间的均值μ 和 方差logσ (latent_channels默认4，LDM标准)
        self.out_channels = 2048  # ResNet50最终特征通道
        self.conv_mu = nn.Conv2d(self.out_channels, latent_channels, 1)
        self.conv_logsigma = nn.Conv2d(self.out_channels, latent_channels, 1)

    def forward(self, x):
        # x: (B, 1, 512, 512) → 输出: (B, latent_channels, 16, 16)
        feat = self.backbone(x)  # 下采样32倍：512→16
        mu = self.conv_mu(feat)
        logsigma = self.conv_logsigma(feat)
        return mu, logsigma


class Decoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        # 对称ResNet编码器，上采样32倍：16→512
        self.init_conv = nn.Conv2d(latent_channels, 2048, 1)

        # 上采样模块：转置卷积（逐步放大特征图）
        self.upsample = nn.Sequential(
            # 16→32
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024), nn.ReLU(),
            # 32→64
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512), nn.ReLU(),
            # 64→128
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            # 128→256
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            # 256→512
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            # 最终输出：单通道CT图像
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()  # 输出[-1,1]，匹配CT预处理
        )

    def forward(self, z):
        # z: 潜在向量 (B, latent_channels,16,16) → 输出 (B,1,512,512)
        x = self.init_conv(z)
        return self.upsample(x)

class CT_LDM_VAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.encoder = ResNetEncoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def reparameterize(self, mu, logsigma):
        """重参数化：LDM采样潜在向量z"""
        sigma = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, x):
        # 编码：CT → 潜在空间μ,logσ
        mu, logsigma = self.encoder(x)
        # 采样潜在向量z
        z = self.reparameterize(mu, logsigma)
        # 解码：z → 重建CT
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma

# ===================== 你的VAE模型（不变）=====================
model = CT_LDM_VAE(latent_channels=4)
model.eval()

# ===================== 🔥 修复后的输入流程 =====================
# 1. 读取你的DCM CT文件（替换为你的dcm文件路径）
dcm_path = r"G:\PythonProject\MSU_CT_Denoising\data\raw\Raw_Brain_Stroke_CT_Dataset\Bleeding\DICOM\10033.dcm"  # 这里改路径！
ct_hu = read_ct_dcm(dcm_path)

# 2. 预处理（自动修复numpy→张量，无报错）
ct_input = ct_preprocess(ct_hu)

# 3. 添加batch维度 (模型要求：[Batch, Channel, H, W])
ct_input = ct_input.unsqueeze(0)  # (1,1,512,512)

# 4. 推理（正常运行）
with torch.no_grad():
    recon_ct, mu, logsigma = model(ct_input)

# 验证输出
print("输入形状:", ct_input.shape)       # torch.Size([1, 1, 512, 512])
print("重建形状:", recon_ct.shape)      # torch.Size([1, 1, 512, 512])

import matplotlib.pyplot as plt
ct_img = ct_input.squeeze().cpu().numpy()
recon_ct_img = recon_ct.squeeze().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ct_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(recon_ct_img, cmap='gray')
plt.tight_layout()
plt.show()

