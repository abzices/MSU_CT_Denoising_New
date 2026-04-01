import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
import glob
import pydicom
from src.utils.DicomConvertion import dicom_to_numpy
from src.data.NoiseSimulate import LowDoseCTSimulator
import pywt


@dataclass
class Config:
    # 文件路径
    root_full_dose : str
    root_quarter_dose : str
    output_dir : str

    # 预处理参数
    hu_min : float = -1000.0
    hu_max : float = 400.0

    # 随机种子
    random_seed :int = 42

    # 目标文件夹末位标识符
    end_mark : str = '_1_Sharp'

    # 仿真低剂量数值，值取0~1，值越小则加噪越明显
    dose_level : float = 0.1

    # 小波变换基
    wavelet_base : str = 'haar'

root_full_dose = r"./data/raw/Full Dose"
root_quarter_dose = r"./data/raw/Quarter Dose"
output_dir = r"./data/Processed/AAPM_Dataset"
config = Config(root_full_dose=root_full_dose,
                root_quarter_dose=root_quarter_dose,
                output_dir=output_dir)

# 定义用于数据预处理的数据集类
class AAPMDataset(Dataset):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.root_full_dose = config.root_full_dose
        self.root_quarter_dose = config.root_quarter_dose
        self.output_dir = config.output_dir
        self.end_mark = config.end_mark

        # 验证母文件夹是否存在
        if not os.path.exists(config.root_full_dose) or not os.path.exists(config.root_quarter_dose):
            raise FileNotFoundError()

        # 收集所有.IMA文件的路径
        self.file_paths = []
        # 查找所有以 end_mark 结尾的子文件夹
        # 使用 glob 匹配模式：full_dose_dir/* + end_mark
        pattern = os.path.join(self.root_full_dose, f"*{self.end_mark}")
        subdirs = glob.glob(pattern)

        for subdir in subdirs:
            # 每个子文件夹中查找所有 .IMA 文件（不区分大小写）
            ima_files = glob.glob(os.path.join(subdir, "*.IMA"))
            self.file_paths.extend(ima_files)

        # 打印数据集大小，便于调试
        print(f"Found {len(self.file_paths)} .IMA files in {len(subdirs)} folders with suffix '{self.end_mark}'.")

    def __convert_to_npy(self, classify):
        """
        输入一个原始文件夹并遍历，有针对性的选取其中的某一类子文件夹，对其下DICOM文件HU值截断并转换为.npy文件保存
        :param: str:待处理的DICOM文件属于正常(NDCT,Normal)/低剂量(LDCT,Low-dose)
        :return:dicom->npy
        """
        # 遍历母文件夹子项
        if classify == "NDCT" or classify == "Normal-dose":
            for item in os.listdir(self.root_full_dose):
                item_path = os.path.join(config.root_full_dose, item)
                # 有针对性的选择某些文件夹
                if os.path.isdir(item_path) and item.endswith(self.end_mark):
                    print(f"开始处理文件夹：{item_path}")
                    # 返回处理后的文件夹下的.npy文件的路径索引序列
                    saved_folder = dicom_to_numpy(folder_path=item_path,
                                                      hu_min=config.hu_min,
                                                      hu_max=config.hu_max,
                                                      output_path=os.path.join(config.output_dir, classify),
                                                      classify=classify)

        if classify == "LDCT" or classify == "Low-dose":
            for item in os.listdir(self.root_quarter_dose):
                item_path = os.path.join(config.root_quarter_dose, item)
                # 有针对性的选择某些文件夹
                if os.path.isdir(item_path) and item.endswith(self.end_mark):
                    print(f"开始处理文件夹：{item_path}")
                    # 返回处理后的文件夹下的.npy文件的路径索引序列
                    saved_folder = dicom_to_numpy(folder_path=item_path,
                                                  hu_min=config.hu_min,
                                                  hu_max=config.hu_max,
                                                  output_path=os.path.join(config.output_dir, classify),
                                                  classify=classify)

        else:
            raise ValueError("classify输入不符合要求，应重新输入！")

    def downgrade_simulation(self):
        """
        模拟车载低剂量CT的物理降级过程，对正常剂量CT添加指定程度的噪声以扩充数据集，向指定文件夹输出仿真Low-dose CT
        """
        simulation = LowDoseCTSimulator(input_dir=self.config.root_full_dose,
                                        output_dir=self.config.output_dir,
                                        dose_level=self.config.dose_level,
                                        end_mark=self.config.end_mark)
        simulation.process_folder_file()

        return f"模拟{config.dose_level}倍低剂量CT已成功生成！"

    def wavelet_downsample(self, input_img, wavelet=config.wavelet_base, output_mode = 0):
        """
        基于小波变换对图像降采样
        :param input_img: 待降采样的图片，只能处理经归一化至[-1,1]的灰度图像
        :param wavelet: 可选小波基
        :param output_mode: 输出模式 0=保留全部4组结果(默认) ; 1=仅保留降采样cA
        :return: 根据output_mode返回对应结果（图像格式：uint8, 0-255）
        """
        # 1. 检查 image 是否为二维数组
        if input_img.ndim != 2:
            raise ValueError("输入图像必须是二维灰度图像，当前维度为 {}".format(input_img.ndim))

        # 2. 检查 image 的值范围是否为 [-1, 1]（允许微小浮点误差）
        eps = 1e-6
        if not (np.all(input_img >= -1 - eps) and np.all(input_img <= 1 + eps)):
            raise ValueError("输入图像必须归一化至 [-1, 1] 范围内")

        # 3. 检查 wavelet 是否有效
        try:
            pywt.Wavelet(wavelet)
        except Exception:
            raise ValueError("无效的小波基名称 '{}'，请使用 pywt.wavelist() 查看可用小波".format(wavelet))

        # 4. 检查 output_mode 是否合法
        if output_mode not in (0, 1):
            raise ValueError("output_mode 必须为 0 或 1，当前值为 {}".format(output_mode))

        # 5. 执行二维离散小波变换（一层）
        coeffs = pywt.dwt2(input_img, wavelet, mode='symmetric')
        cA, (cH, cV, cD) = coeffs

        # 6. 根据 output_mode 返回结果
        if output_mode == 1:
            return cA  # 仅低频分量
        else:  # output_mode == 0
            return cA, cH, cV, cD  # 四个子带

    def process(self):
        pass

    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本。
        Args:
            idx (int): 样本索引
        Returns:
            torch.Tensor: 图像张量，形状为 (1, H, W)
            (可选) str: 文件路径（便于调试）
        """
        # 获取文件路径
        file_path = self.file_paths[idx]

        # 读取 DICOM 文件（假设 .IMA 为 DICOM 格式）
        try:
            ds = pydicom.dcmread(file_path, force=True)
            # 获取像素数组，通常为 (H, W) 的 numpy 数组
            img = ds.pixel_array.astype(np.float32)
        except Exception as e:
            # 如果读取失败，可打印错误并返回空（实际应用中可抛出异常）
            print(f"Error reading {file_path}: {e}")
            raise e

        # 转换为 torch 张量，并增加通道维度 (C, H, W)，这里 C=1
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # shape: (1, H, W)

        # 返回图像张量，也可同时返回文件路径（方便调试）
        return img_tensor, file_path


