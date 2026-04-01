# ResNet VAE 模型使用说明

## 概述

本项目实现了一个基于ResNet架构的变分自编码器（VAE），专门用于处理CT图像的去噪任务。模型接受4通道256×256的图像（由1×512×512的CT图像经小波变换得到），将其压缩到潜在空间（8×8×8），然后再重建回4通道256×256。

## 模型架构

### 编码器（Encoder）
- **输入**: 4通道256×256图像
- **结构**: 基于ResNet50或ResNet18架构
- **输出**: 潜在空间表示（均值μ和对数方差logvar），尺寸为8×8×8

### 解码器（Decoder）
- **输入**: 潜在空间表示（8×8×8）
- **结构**: 对称的ResNet解码器，使用转置卷积进行上采样
- **输出**: 重建的4通道256×256图像

## 文件结构

```
src/models/
├── resnet_fromzero.py      # ResNet VAE模型定义
├── BasicBlock.py           # ResNet基础块
├── Bottleneck.py           # ResNet瓶颈块
├── ResNetEncoder.py        # 编码器
└── ResNetDecoder.py        # 解码器

train_resnet_vae.py         # 训练脚本
dataset.py                  # 数据集加载
```

## 模型特点

### 1. 两种模型版本
- **ResNet50Autoencoder**: 完整版，约150M参数
- **ResNet18Autoencoder**: 轻量级版本，约24M参数

### 2. VAE特性
- 重参数化技巧确保梯度可回传
- KL散度损失用于正则化潜在空间
- 支持训练和推理模式切换

### 3. 训练策略
- **阶段1**: 冻结编码器，仅训练解码器
- **阶段2**: 解冻所有层，微调整个模型
- **KL退火**: 动态调整KL损失权重

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision numpy tqdm tensorboard
```

### 2. 准备数据

确保数据集按照以下结构组织：
```
data/Processed/AAPM_Dataset/
├── NDCT_DWT_PT/    # 正常剂量CT的小波变换结果
└── LDCT_DWT_PT/    # 低剂量CT的小波变换结果
```

### 3. 测试模型

```bash
python src/self_models/resnet_fromzero.py
```

这将测试模型的输入输出尺寸和参数量。

### 4. 训练模型

```bash
python train_resnet_vae.py
```

## 配置说明

在 `train_resnet_vae.py` 中可以修改以下配置：

```python
@dataclass
class TrainConfig:
    # 模型选择
    model_type: str = "resnet50"  # 可选: "resnet50" 或 "resnet18"
    latent_channels: int = 8      # 潜在空间通道数
    
    # 训练超参
    batch_size: int = 8
    num_epochs_stage1: int = 20   # 阶段1训练轮数
    num_epochs_stage2: int = 50   # 阶段2训练轮数
    lr_stage1: float = 1e-4       # 阶段1学习率
    lr_stage2: float = 5e-5       # 阶段2学习率
    
    # 损失权重
    l1_weight: float = 1.0
    mse_weight: float = 1.0
    high_freq_weight: float = 2.0  # 高频通道重建损失权重
    kl_weight_max: float = 1e-3    # KL损失最大权重
```

## 模型使用示例

### 基本使用

```python
import torch
from src.self_models.resnet_fromzero import ResNet50Autoencoder

# 初始化模型
model = ResNet50Autoencoder(latent_channels=8)
model.eval()

# 准备输入
x = torch.randn(1, 4, 256, 256)  # batch_size=1, 4通道, 256x256

# 前向传播
with torch.no_grad():
    recon, mu, logvar = model(x)

print(f"输入尺寸: {x.shape}")  # torch.Size([1, 4, 256, 256])
print(f"潜在空间尺寸: {mu.shape}")  # torch.Size([1, 8, 8, 8])
print(f"重建输出尺寸: {recon.shape}")  # torch.Size([1, 4, 256, 256])
```

### 加载预训练模型

```python
import torch
from src.self_models.resnet_fromzero import ResNet50Autoencoder

# 初始化模型
model = ResNet50Autoencoder(latent_channels=8)

# 加载权重
checkpoint = torch.load('models/resnet_stage2_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 使用模型进行推理
with torch.no_grad():
    recon, mu, logvar = model(input_tensor)
```

## 训练监控

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir=./runs/resnet_vae
```

在浏览器中打开 `http://localhost:6006` 查看：
- 训练和验证损失曲线
- 重建图像可视化
- KL权重变化
- 学习率调度

## 模型性能

### ResNet50Autoencoder
- **参数量**: ~150M
- **潜在空间**: 8×8×8
- **输入/输出**: 4×256×256
- **特点**: 强大的特征提取能力，适合复杂任务

### ResNet18Autoencoder
- **参数量**: ~24M
- **潜在空间**: 8×8×8
- **输入/输出**: 4×256×256
- **特点**: 轻量级，训练速度快，适合资源受限环境

## 损失函数

模型使用组合损失函数：

```
Total Loss = L1 Loss + MSE Loss + KL Loss + High Frequency Loss
```

- **L1 Loss**: 像素级重建损失
- **MSE Loss**: 像素级重建损失
- **KL Loss**: 潜在空间正则化
- **High Frequency Loss**: 专门针对小波变换的高频通道（LH/HL/HH）

## 注意事项

1. **输入格式**: 确保输入是4通道256×256的张量
2. **训练策略**: 建议先进行阶段1训练，再进行阶段2微调
3. **KL退火**: KL损失权重会从0逐渐增加到最大值，避免训练初期的不稳定
4. **梯度裁剪**: 训练过程中使用梯度裁剪（max_norm=1.0）防止梯度爆炸
5. **学习率调度**: 阶段2使用ReduceLROnPlateau调度器自动调整学习率

## 常见问题

### Q: 如何调整潜在空间大小？
A: 修改 `latent_channels` 参数。例如，设置为16可以得到8×16×16的潜在空间。

### Q: 训练时显存不足怎么办？
A: 可以：
1. 减小 `batch_size`
2. 使用ResNet18代替ResNet50
3. 减小 `latent_channels`

### Q: 如何使用预训练的ResNet权重？
A: 当前实现是从零开始训练。如果需要使用预训练权重，可以修改编码器的初始化部分，参考 `resnet_vae.py` 中的实现。

### Q: 模型推理时输出不稳定？
A: 确保在推理时调用 `model.eval()`，并使用 `torch.no_grad()` 上下文管理器。

## 引用

如果这个模型对你的研究有帮助，请考虑引用：

```bibtex
@software{resnet_vae_ct_denoising,
  title={ResNet VAE for CT Image Denoising},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MSU_CT_Denoising_New}
}
```

## 许可证

本项目遵循MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- Email: your.email@example.com
- GitHub Issues: https://github.com/yourusername/MSU_CT_Denoising_New/issues
