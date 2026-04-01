import numpy as np
import astra
import matplotlib.pyplot as plt


class LowDoseCTSimulatorASTRA:
    """
    基于 ASTRA Toolbox 的低剂量 CT 模拟器。

    该类能够将高剂量 CT 图像投影至正弦域，依据文献提出的复合泊松-高斯噪声模型
    (Compound Poisson-Gaussian noise model) 进行降剂量模拟，并将其重建回图像域。
    """

    def __init__(self, I0_high=1e6, sigma_e=10.0, m_e=0.0,
                 num_angles=360, use_cuda=True, value_scale=0.02):
        """
        初始化模拟器参数
        """
        self.I0_high = I0_high
        self.sigma_e = sigma_e
        self.m_e = m_e
        self.num_angles = num_angles
        self.use_cuda = use_cuda
        self.value_scale = value_scale

        self.vol_geom = None
        self.proj_geom = None

        # 算法名称
        self.fp_algo = 'FP_CUDA' if self.use_cuda else 'FP'
        self.fbp_algo = 'FBP_CUDA' if self.use_cuda else 'FBP'

    def _init_astra_geometries(self, image_shape):
        """根据输入图像的尺寸初始化 ASTRA 体几何和投影几何 (平行束)"""
        if len(image_shape) != 2:
            raise ValueError("当前仅支持 2D 图像输入。")

        nx, ny = image_shape
        self.vol_geom = astra.create_vol_geom(nx, ny)

        # 探测器尺寸需能覆盖图像对角线
        num_detectors = int(np.ceil(np.sqrt(nx ** 2 + ny ** 2)))
        angles = np.linspace(0, np.pi, self.num_angles, endpoint=False)

        # 创建平行束投影几何
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, num_detectors, angles)

    def forward_project(self, image):
        """图像域 -> 投影域 (无噪正弦图)"""
        if self.vol_geom is None or self.proj_geom is None:
            self._init_astra_geometries(image.shape)

        scaled_image = image * self.value_scale

        img_id = astra.data2d.create('-vol', self.vol_geom, scaled_image)
        sino_id = astra.data2d.create('-sino', self.proj_geom)

        cfg = astra.astra_dict(self.fp_algo)
        cfg['VolumeDataId'] = img_id
        cfg['ProjectionDataId'] = sino_id

        # 修复：CPU 算法强制需要 ProjectorId
        proj_id = None
        if not self.use_cuda:
            # 使用 'line' 模型创建投影器
            proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom)
            cfg['ProjectorId'] = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        sino_clean = astra.data2d.get(sino_id)

        # 清理内存
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(img_id)
        astra.data2d.delete(sino_id)
        if proj_id is not None:
            astra.projector.delete(proj_id)

        return sino_clean

    def inject_noise(self, sino_clean, dose_ratio):
        """在投影域注入泊松-高斯噪声"""
        I0_low = self.I0_high * dose_ratio

        #print(np.max(sino_clean))
        transmission = np.exp(-sino_clean)
        I_low_clean = I0_low * transmission

        I_low_poisson = np.random.poisson(I_low_clean).astype(np.float32)
        I_low_gaussian = np.random.normal(loc=self.m_e, scale=self.sigma_e, size=sino_clean.shape)

        I_low_noisy = I_low_poisson + I_low_gaussian

        # 防止极端光子饥饿截断，防止 log 报错
        I_low_noisy = np.clip(I_low_noisy, a_min=1e-5, a_max=None)

        sino_noisy = np.log(I0_low / I_low_noisy)
        return sino_noisy

    def reconstruct(self, sino):
        """投影域 -> 图像域 (滤波反投影重建)"""
        sino_id = astra.data2d.create('-sino', self.proj_geom, sino)
        rec_id = astra.data2d.create('-vol', self.vol_geom)

        cfg = astra.astra_dict(self.fbp_algo)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id

        # 修复：CPU 算法强制需要 ProjectorId
        proj_id = None
        if not self.use_cuda:
            proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom)
            cfg['ProjectorId'] = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        rec_image = astra.data2d.get(rec_id)

        # 清理内存
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.data2d.delete(rec_id)
        if proj_id is not None:
            astra.projector.delete(proj_id)

        return rec_image / self.value_scale

    def simulate(self, image, dose_ratio):
        """流水线化：执行完整的 投影 -> 加噪 -> 重建 流程。"""
        sino_clean = self.forward_project(image)
        sino_noisy = self.inject_noise(sino_clean, dose_ratio)
        rec_noisy = self.reconstruct(sino_noisy)

        return rec_noisy, sino_noisy, sino_clean


# ==========================================
# 测试与使用示例
# ==========================================
if __name__ == "__main__":
    from skimage.data import shepp_logan_phantom

    # 1. 生成高剂量图像作为 Ground Truth (缩放到[0, 1] 范围)
    high_dose_image = shepp_logan_phantom()

    # 2. 初始化模拟器 (现在 use_cuda=False 也能完美运行了)
    simulator = LowDoseCTSimulatorASTRA(
        I0_high=1e6,
        sigma_e=10.0,
        num_angles=360,
        use_cuda=False
    )

    # 3. 模拟不同剂量的低剂量图像
    dose_10_percent = 0.10
    dose_05_percent = 0.05

    recon_10_noisy, sino_10_noisy, sino_clean = simulator.simulate(high_dose_image, dose_ratio=dose_10_percent)
    recon_05_noisy, sino_05_noisy, _ = simulator.simulate(high_dose_image, dose_ratio=dose_05_percent)

    # 4. 绘图对比展示
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：图像域
    axes[0, 0].imshow(high_dose_image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('High-Dose Ground Truth Image')
    axes[0, 1].imshow(recon_10_noisy, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Low-Dose CT (10% Dose)')
    axes[0, 2].imshow(recon_05_noisy, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Low-Dose CT (5% Dose)')

    # 第二行：投影域(正弦图)
    axes[1, 0].imshow(sino_clean, cmap='gray', aspect='auto')
    axes[1, 0].set_title('Clean Sinogram')
    axes[1, 1].imshow(sino_10_noisy, cmap='gray', aspect='auto')
    axes[1, 1].set_title('Noisy Sinogram (10%)')
    axes[1, 2].imshow(sino_05_noisy, cmap='gray', aspect='auto')
    axes[1, 2].set_title('Noisy Sinogram (5%)')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()