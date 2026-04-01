import os
import glob
import torch
import pywt
import numpy as np
from tqdm import tqdm

def batch_convert_to_dwt_pt(src_dir, tgt_dir, wavelet='haar'):
    """
    批量将 numpy 文件转换为离散小波变换 (DWT) 后的 PyTorch 张量并保存为 .pt 文件。

    参数:
    src_dir (str): 源目录，包含要转换的 .npy 文件。
    tgt_dir (str): 目标目录，用于保存转换后的 .pt 文件。
    wavelet (str, 可选): 使用的小波类型，默认为 'haar'。

    返回:
    None
    """
    os.makedirs(tgt_dir, exist_ok=True)
    files = glob.glob(os.path.join(src_dir, "*.npy"))

    print(f"开始转换 {src_dir} -> {tgt_dir} (共 {len(files)} 个文件)")
    for f in tqdm(files):
        # 1. 读取原始 numpy 数据
        img_np = np.load(f).astype(np.float32).squeeze()

        # 2. 进行 DWT 变换
        coeffs = pywt.dwt2(img_np, wavelet, mode='symmetric')
        cA, (cH, cV, cD) = coeffs

        # 3. 堆叠并除以 2.0 归一化到 [-1, 1]
        dwt_stacked = np.stack([cA, cH, cV, cD], axis=0) / 2.0

        # 4. 转换为 Tensor 并保存为 .pt 文件
        tensor = torch.from_numpy(dwt_stacked).float()

        # 保存 (替换后缀名为 .pt)
        save_name = os.path.basename(f).replace(".npy", ".pt")
        torch.save(tensor, os.path.join(tgt_dir, save_name))


if __name__ == "__main__":
    # 原始 .npy 文件夹
    raw_ndct = r"G:\MSU_CT_Denoising_New\data\Processed\AAPM_Dataset\NDCT"
    raw_ldct = r"G:\MSU_CT_Denoising_New\data\Processed\AAPM_Dataset\LDCT"

    # 新的存放 DWT .pt 的文件夹
    dwt_ndct = r"G:\MSU_CT_Denoising_New\data\Processed\AAPM_Dataset\NDCT_DWT_PT"
    dwt_ldct = r"G:\MSU_CT_Denoising_New\data\Processed\AAPM_Dataset\LDCT_DWT_PT"

    batch_convert_to_dwt_pt(raw_ndct, dwt_ndct)
    batch_convert_to_dwt_pt(raw_ldct, dwt_ldct)
    print("全部 DWT 特征离线提取完成！")