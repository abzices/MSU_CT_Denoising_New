import numpy as np
import astra
import matplotlib.pyplot as plt
import pydicom
import os
from tqdm import tqdm

class LowDoseCTSimulator:
    """
    基于 ASTRA Toolbox 的低剂量 CT 物理仿真器。
    模拟物理过程：图像(HU) -> 线性衰减系数(μ) -> 弦图(Sinogram) -> 泊松-高斯噪声 -> FBP重建 -> 图像(HU)
    """
    def __init__(self, input_dir, output_dir, dose_level, gaussian_sigma=10.0, I0=1e6, end_mark='_1_Sharp', use_gpu=True):
        """
        定义参数
        :param input_dir: 目标正常剂量的文件夹
        :param output_dir: 目标输出文件夹
        :param dose_level: 与正常剂量水平之比率，越接近0，则噪声越显著
        :param gaussian_sigma: 高斯电子噪声方差
        :param I0: 正常剂量下的入射光子数，设为常数
        :param end_mark: str:指定文件夹的末位标识字符串
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dose_level = dose_level
        self.gaussian_sigma = gaussian_sigma
        self.I0 = I0
        self.end_mark = end_mark
        self.use_gpu = use_gpu  # 建议设为 True

    def load_dicom_as_numpy(self, file_path):
        """读取DICOM并转换为物理单位"""
        ds = pydicom.dcmread(file_path, force=True)
        ds_pixel = ds.pixel_array.astype(np.float32)

        # 安全获取 Slope 和 Intercept
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)

        ds_HU = ds_pixel * slope + intercept

        # 将 HU 转换为线性衰减系数 μ (假设120kVp下水的 μ ≈ 0.194 cm^-1)
        ds_miu = 0.194 * (ds_HU / 1000.0 + 1.0)
        ds_miu = np.clip(ds_miu, 0, None)

        # 提取物理尺寸并转为 cm
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing_cm = float(ds.PixelSpacing[0]) / 10.0
        else:
            pixel_spacing_cm = 0.1  # 默认 1mm

        return ds_miu, ds_HU, pixel_spacing_cm

    def add_compound_poisson_gaussian(self, clean_sinogram):
        """在正弦图域添加复合泊松-高斯噪声"""
        I0_low = self.I0 * self.dose_level

        try:
            # Beer-Lambert 定律：传输率 = exp(-积分衰减)
            transmission = np.exp(-clean_sinogram)
            I_low_clean = I0_low * transmission

            # 泊松噪声(模拟光子统计涨落)
            I_low_poisson = np.random.poisson(I_low_clean).astype(np.float32)
            # 高斯噪声(模拟探测器电子背景噪声)
            I_low_gaussian = np.random.normal(0, self.gaussian_sigma, size=clean_sinogram.shape)

            I_low_noisy = I_low_poisson + I_low_gaussian

            # 防止极端光子饥饿截断，防止 log 报错
            I_low_noisy = np.clip(I_low_noisy, a_min=0.1, a_max=None)

            noisy_sinogram = np.log(I0_low / I_low_noisy)

            return noisy_sinogram
        except Exception as e:
            print(f"添加噪声时发生错误: {e}")

    def process_folder_file(self):
        """批量处理流水线"""
        # 修复：确保输出总文件夹存在
        os.makedirs(self.output_dir, exist_ok=True)

        for folder_name in os.listdir(self.input_dir):
            folder_path = os.path.join(self.input_dir, folder_name)
            if os.path.isdir(folder_path) and folder_path.endswith(self.end_mark):
                print(f"🚀 开始处理文件夹：{folder_name}")

                # 创建对应的输出子文件夹
                save_dir = os.path.join(self.output_dir, f"Level_{self.dose_level}_Low-dose_Simulation")
                os.makedirs(save_dir, exist_ok=True)

                file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.dcm', '.ima'))]
                for item in tqdm(file_list, desc=f"🚀 正在仿真 {folder_name}", unit="张"):
                    if item.lower().endswith(('.dcm', '.ima')):
                        file_full_path = os.path.join(folder_path, item)
                        img_miu, original_HU, pixel_spacing_cm = self.load_dicom_as_numpy(file_full_path)
                        rows, cols = img_miu.shape

                        # 统一缩放至物理积分长度
                        img_scaled = img_miu * pixel_spacing_cm

                        # 几何定义
                        vol_geom = astra.create_vol_geom(rows, cols)

                        # 物理防伪影：角度数最好比像素维度略多
                        num_angles = int(np.pi * max(rows, cols) / 2)
                        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
                        det_count = int(np.ceil(np.sqrt(rows ** 2 + cols ** 2)))
                        proj_geom = astra.create_proj_geom('parallel', 1.0, det_count, angles)

                        # 初始化所有 ASTRA 内存指针为 None，防止 finally 报错
                        vol_id, sino_id, rec_id = None, None, None
                        alg_fp_id, alg_fbp_id = None, None
                        noisy_sino_data_id, proj_id = None, None

                        try:
                            if self.use_gpu:
                                # ==========================================
                                # 🚀 GPU 模式 (纯 CUDA 算法)
                                # ==========================================
                                # 1. 正向投影 (图像 -> 弦图)，使用 FP_CUDA 算法
                                vol_id = astra.data2d.create('-vol', vol_geom, img_scaled)
                                sino_id = astra.data2d.create('-sino', proj_geom, 0.0)

                                cfg_fp = astra.astra_dict('FP_CUDA')
                                cfg_fp['VolumeDataId'] = vol_id
                                cfg_fp['ProjectionDataId'] = sino_id
                                alg_fp_id = astra.algorithm.create(cfg_fp)
                                astra.algorithm.run(alg_fp_id)

                                sinogram = astra.data2d.get(sino_id)

                                # 2. 注入泊松-高斯噪声
                                noisy_sinogram = self.add_compound_poisson_gaussian(sinogram)

                                # 3. 反向滤波投影 (弦图 -> 图像)，使用 FBP_CUDA 算法
                                noisy_sino_data_id = astra.data2d.create('-sino', proj_geom, noisy_sinogram)
                                rec_id = astra.data2d.create('-vol', vol_geom, 0.0)

                                cfg_fbp = astra.astra_dict('FBP_CUDA')
                                cfg_fbp['ReconstructionDataId'] = rec_id
                                cfg_fbp['ProjectionDataId'] = noisy_sino_data_id
                                alg_fbp_id = astra.algorithm.create(cfg_fbp)
                                astra.algorithm.run(alg_fbp_id)

                                recon_img = astra.data2d.get(rec_id)

                            else:
                                # ==========================================
                                # 🐢 CPU 模式 (基于 Projector)
                                # ==========================================
                                proj_id = astra.create_projector('line', proj_geom, vol_geom)
                                sino_id, sinogram = astra.create_sino(img_scaled, proj_id)

                                noisy_sinogram = self.add_compound_poisson_gaussian(sinogram)

                                noisy_sino_data_id = astra.data2d.create('-sino', proj_geom, noisy_sinogram)
                                rec_id = astra.data2d.create('-vol', vol_geom, 0.0)

                                cfg_fbp = astra.astra_dict('FBP')
                                cfg_fbp['ReconstructionDataId'] = rec_id
                                cfg_fbp['ProjectionDataId'] = noisy_sino_data_id
                                cfg_fbp['ProjectorId'] = proj_id

                                alg_fbp_id = astra.algorithm.create(cfg_fbp)
                                astra.algorithm.run(alg_fbp_id)

                                recon_img = astra.data2d.get(rec_id)

                            # 还原回 HU 值
                            recon_miu = recon_img / pixel_spacing_cm
                            recon_HU = (recon_miu / 0.194 - 1.0) * 1000.0

                            # 保存结果
                            item_basename = os.path.splitext(os.path.basename(item))[0]
                            output_path = os.path.join(save_dir, item_basename + ".npy")
                            np.save(output_path, recon_HU)

                        finally:
                            # 极致安全地释放内存（对于批量处理极为重要）
                            for ast_id in [alg_fp_id, alg_fbp_id]:
                                if ast_id is not None:
                                    astra.algorithm.delete(ast_id)
                            for ast_id in [vol_id, sino_id, noisy_sino_data_id, rec_id]:
                                if ast_id is not None:
                                    astra.data2d.delete(ast_id)
                            if proj_id is not None:
                                astra.projector.delete(proj_id)

if __name__ == "__main__":
    # 配置你的路径
    input_dir = r"G:\PythonProject\MSU_CT_Denoising\notebooks\111"
    output_dir = r"G:\PythonProject\MSU_CT_Denoising\notebooks\Simulation"

    # 实例化并运行
    simu = LowDoseCTSimulator(
        input_dir=input_dir,
        output_dir=output_dir,
        dose_level=0.1,
        use_gpu=True,  # 如果报错没有cuda，改为False，但会非常慢
        end_mark='_1_Sharp'
    )
    # 取消注释以运行批量仿真
    #simu.process_folder_file()


