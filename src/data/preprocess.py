import os
import numpy as np
from dataclasses import dataclass

from src.utils.DicomConvertion import dicom_to_numpy

@dataclass
class Config:
    # 文件路径
    root_full_dose: str = r"G:\PythonProject\MSU_CT_Denoising\data\raw\Full Dose"
    root_quarter_dose: str = r"G:\PythonProject\MSU_CT_Denoising\data\raw\Quarter Dose"
    output_dir: str = r"G:\PythonProject\MSU_CT_Denoising\data\Processed\AAPM_Dataset"

    # 预处理参数
    hu_min: float = -1000.0
    hu_max: float = 400.0

    # 目标文件夹末位标识符
    end_mark: str = '_1_Sharp'

def convert_to_npy(config, classify):
    """
    输入一个原始文件夹并遍历，有针对性的选取其中的某一类子文件夹，对其下DICOM文件HU值截断并转换为.npy文件保存
    :param: str:待处理的DICOM文件属于正常(NDCT,Normal)/低剂量(LDCT,Low-dose)
    :return:dicom->npy
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # 遍历母文件夹子项
    if classify == "NDCT" or classify == "Normal-dose":
        root_path = config.root_full_dose
    elif classify == "LDCT" or classify == "Low-dose":
        root_path = config.root_quarter_dose
    else:
        raise ValueError("classify输入不符合要求，应重新输入！")

    # 统一遍历逻辑
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        # 有针对性的选择某些文件夹
        if os.path.isdir(item_path) and item.endswith(config.end_mark):
            print(f"开始处理文件夹：{item_path}")
            # 关键修改：output_path 只拼接一级 classify
            saved_folder = dicom_to_numpy(
                folder_path=item_path,
                hu_min=config.hu_min,
                hu_max=config.hu_max,
                output_path=os.path.join(config.output_dir, classify),
                classify=classify
            )
    print(f"{classify} 类型文件处理完成！")

if __name__ == "__main__":
    config = Config()
    convert_to_npy(config=config, classify="NDCT")
    convert_to_npy(config=config, classify="LDCT")