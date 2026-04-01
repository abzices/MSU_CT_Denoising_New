import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
import glob
import pywt

@dataclass
class Config:
    # 读取你【预处理后】的 .npy 文件夹，而不是原始文件夹
    # 先用data.preprocess把 DICOM 转成归一化到 [-1, 1] 的 .npy
    ndct_dir: str = r"./data/Processed/AAPM_Dataset/NDCT"
    ldct_dir: str = r"./data/Processed/AAPM_Dataset/LDCT"

    # 小波变换基
    wavelet_base: str = 'haar'

# ==========================================
# 标准深度学习 Dataset 类
# 职责：读取配对的归一化.npy，做DWT，转Tensor
# ==========================================
class AAPMDenoisingDataset(Dataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # 1. 验证文件夹是否存在
        if not os.path.exists(self.config.ndct_dir) or not os.path.exists(self.config.ldct_dir):
            raise FileNotFoundError("找不到处理后的数据文件夹，请先运行数据转换脚本！")

        # 2. 收集所有 .npy 文件路径，并进行配对排序
        # 假设你的转换脚本保证了同源图像的 NDCT 和 LDCT 文件名是一致的
        ndct_files = sorted(glob.glob(os.path.join(self.config.ndct_dir, "*.npy")))
        ldct_files = sorted(glob.glob(os.path.join(self.config.ldct_dir, "*.npy")))

        # 严格检查配对数量
        assert len(ndct_files) == len(ldct_files), \
            f"数据不匹配！NDCT有{len(ndct_files)}张，LDCT有{len(ldct_files)}张。"
        assert len(ndct_files) > 0, "文件夹中没有找到 .npy 文件！"

        self.ndct_paths = ndct_files
        self.ldct_paths = ldct_files

        print(f"Dataset 初始化成功！共找到 {len(self.ndct_paths)} 对图像。")

    def _dwt_to_tensor(self, img_np):
        """
        内部辅助函数：对 [-1, 1] 的 numpy 图像做 DWT，并转换为 4通道 Tensor
        """
        # img_np shape: (H, W)
        coeffs = pywt.dwt2(img_np, self.config.wavelet_base, mode='symmetric')
        cA, (cH, cV, cD) = coeffs

        # 将四个子带堆叠为通道维度: shape (4, H/2, W/2)
        dwt_stacked = np.stack([cA, cH, cV, cD], axis=0)

        # 转换为 PyTorch Tensor (float32是网络必须的格式)
        return torch.from_numpy(dwt_stacked).float()

    def __len__(self):
        return len(self.ndct_paths)

    def __getitem__(self, idx):
        """
        返回配对的 4 通道频域 Tensor
        """
        # 1. 加载预处理好的归一化 [-1, 1] 的 numpy 数组
        ndct_np = np.load(self.ndct_paths[idx]).astype(np.float32)
        ldct_np = np.load(self.ldct_paths[idx]).astype(np.float32)

        # 2. 检查维度 (如果是 3D 的话，去掉可能多余的 channel 维度)
        if ndct_np.ndim > 2:
            ndct_np = ndct_np.squeeze()
            ldct_np = ldct_np.squeeze()

        # 3. 进行二维离散小波变换并转为 Tensor -> shape: (4, 256, 256)
        ndct_tensor = self._dwt_to_tensor(ndct_np)
        ldct_tensor = self._dwt_to_tensor(ldct_np)

        # 返回 脏图(输入), 干净图(目标), 方便调试的文件名
        return ldct_tensor, ndct_tensor, os.path.basename(self.ndct_paths[idx])

if __name__ == "__main__":
    # 测试你的 Dataset
    dataset = AAPMDenoisingDataset(Config())

    import matplotlib.pyplot as plt

    # 从 Dataset 拿取第一对图像
    ldct, ndct, name = dataset[0]

    # 通道对应的物理意义
    titles = ['LL (Low Freq / Structure)',
              'LH (High Freq / Horizontal Edge)',
              'HL (High Freq / Vertical Edge)',
              'HH (High Freq / Diagonal Edge)']

    plt.figure(figsize=(16, 4))
    plt.suptitle(f"Sample: {name} | Tensor Shape: {ldct.shape}", fontsize=16)

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        # 取出其中一个通道并转回 numpy
        img_channel = ndct[i].numpy()

        # 画图，使用灰度图。
        # 注意：高频通道(LH, HL, HH)的值绝大部分在 0 附近，所以可能看起来偏暗/灰
        plt.imshow(img_channel, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
