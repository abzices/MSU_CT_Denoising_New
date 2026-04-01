import pydicom
import os
import numpy as np


def dicom_to_numpy(folder_path, hu_min, hu_max, output_path, classify="NDCT"):
    """
    输入一个文件夹，遍历读取所有DICOM文件（.dcm 或 .IMA），转换为HU并截断，保存为.npy文件并输出至指定文件夹。
    :param folder_path: 存放DICOM的文件夹
    :param hu_min: HU值截断下限
    :param hu_max: 截断上限
    :param output_path: 输出文件夹
    :param classify: 处理对象属于正常/低剂量的分类
    """

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    # 创建保存npy的目录
    os.makedirs(output_path, exist_ok=True)

    save_path = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            # 后缀检查：跳过非DICOM文件
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.dcm', '.ima']:
                print(f"跳过非DICOM文件：{filename}")
                continue

            try:
                ds = pydicom.dcmread(file_path)

                # HU值转换
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    slope = ds.RescaleSlope
                    intercept = ds.RescaleIntercept
                    hu_image = ds.pixel_array.astype(np.float32) * slope + intercept
                else:
                    print(f"警告: {filename} 缺少Rescale参数，直接使用像素值")
                    hu_image = ds.pixel_array.astype(np.float32)

                # HU截断 + 归一化
                hu_clipped = np.clip(hu_image, hu_min, hu_max)
                hu_norm = (hu_clipped - hu_min) / (hu_max - hu_min)  # 优化归一化（固定范围，更稳定）
                hu_norm = (2 * hu_norm - 1).astype(np.float32)

                # 输出路径
                out_filename = os.path.splitext(filename)[0] + ".npy"
                out_path = os.path.join(output_path, out_filename)

                # 保存文件
                np.save(out_path, hu_norm)
                save_path.append(out_path)

            except Exception as e:
                print(f"处理失败 {file_path}：{str(e)}")

    return save_path