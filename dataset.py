import os
import glob
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class Config:
    # 路径为经小波变换的图像文件夹！
    ndct_dir: str = r"./data/Processed/AAPM_Dataset/NDCT_DWT_PT"
    ldct_dir: str = r"./data/Processed/AAPM_Dataset/LDCT_DWT_PT"



class AAPMDenoisingDataset(Dataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        ndct_files = sorted(glob.glob(os.path.join(self.config.ndct_dir, "*.pt")))
        ldct_files = sorted(glob.glob(os.path.join(self.config.ldct_dir, "*.pt")))

        assert len(ndct_files) == len(ldct_files), "数据配对数量不一致！"
        self.ndct_paths = ndct_files
        self.ldct_paths = ldct_files
        print(f"Dataset 初始化成功！共找到 {len(self.ndct_paths)} 对张量。")

    def __len__(self):
        return len(self.ndct_paths)

    def __getitem__(self, idx):
        # 💡 Dataloader 现在只做一件事：从硬盘瞬间把 PT 张量拉进内存！
        ndct_tensor = torch.load(self.ndct_paths[idx], weights_only=True)
        ldct_tensor = torch.load(self.ldct_paths[idx], weights_only=True)

        return ldct_tensor, ndct_tensor, os.path.basename(self.ndct_paths[idx])