import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats


class NoiseAnalysis:
    def __init__(self, clean_img_folder_path, noisy_img_folder_path, max_roi_size=(50, 50)):
        self.clean_img_folder_path = clean_img_folder_path
        self.noisy_img_folder_path = noisy_img_folder_path
        self.max_roi_size = max_roi_size

        # 初始化时直接加载并对齐文件列表
        self.clean_files, self.noisy_files = self.__load_npy_file()

    def __load_npy_file(self):
        if not os.path.exists(self.noisy_img_folder_path) or not os.path.exists(self.clean_img_folder_path):
            raise FileNotFoundError("找不到指定的文件夹路径！")

        clean_npy_files = [f for f in os.listdir(self.clean_img_folder_path) if f.endswith(".npy")]
        noisy_npy_files = [f for f in os.listdir(self.noisy_img_folder_path) if f.endswith(".npy")]

        # 严格按照文件名排序，确保一一对应
        clean_npy_files.sort(key=lambda x: os.path.basename(x))
        noisy_npy_files.sort(key=lambda x: os.path.basename(x))

        assert len(clean_npy_files) == len(noisy_npy_files), "两个文件夹中的.npy文件数量不一致！"
        assert len(clean_npy_files) > 0, "文件夹中没有找到.npy文件！"

        return clean_npy_files, noisy_npy_files

    def _normalize_for_display(self, img):
        # 智能判断：如果最大值较小（说明是归一化到[-1,1]的数据），则先还原
        if img.max() <= 2.0:
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        else:
            # 说明是原始 HU 值
            img_clip = np.clip(img, -1000, 400)
            img_norm = cv2.normalize(img_clip, None, 0, 255, cv2.NORM_MINMAX)
        return img_norm.astype(np.uint8)

    def select_roi(self, img):
        print("💡 请在弹出的窗口中拖拽鼠标选择ROI区域！")
        print("👉 按 'SPACE' 或 'ENTER' 确认选区，按 'c' 取消。")
        disp_img = self._normalize_for_display(img)

        roi = cv2.selectROI("Select ROI", disp_img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

        if w == 0 or h == 0:
            raise ValueError("未选择有效的ROI区域！已退出。")

        if w > self.max_roi_size[0] or h > self.max_roi_size[1]:
            print(f"⚠️ 警告: 选择的ROI ({w}x{h}) 超过了最大限制 {self.max_roi_size}。已自动裁剪。")
            center_x, center_y = x + w // 2, y + h // 2
            x = max(0, center_x - self.max_roi_size[0] // 2)
            y = max(0, center_y - self.max_roi_size[1] // 2)
            w, h = self.max_roi_size[0], self.max_roi_size[1]

        return slice(y, y + h), slice(x, x + w)

    def statistical_analysis(self, noise_roi):
        mean = np.mean(noise_roi)
        std_dev = np.std(noise_roi)
        var = np.var(noise_roi)
        skewness = stats.skew(noise_roi.flatten())
        kurtosis = stats.kurtosis(noise_roi.flatten())

        print("\n📊 --- 噪声ROI统计学分析结果 (Noisy - Clean) ---")
        print(f"均值 (Mean):     {mean:.6f}  (理想白噪声应接近0)")
        print(f"标准差 (Std Dev): {std_dev:.6f}  (反映噪声强度)")
        print(f"方差 (Variance):  {var:.6f}")
        print(f"偏度 (Skewness):  {skewness:.6f}  (接近0为对称的正态分布)")
        print(f"峰度 (Kurtosis):  {kurtosis:.6f}  (接近0代表标准高斯尾部)")
        print("-----------------------------------------------\n")

    def power_spectrum_analysis(self, noise_roi):
        f_transform = np.fft.fft2(noise_roi)
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = np.abs(f_shift) ** 2

        h, w = power_spectrum.shape
        center = (h // 2, w // 2)
        y, x = np.indices(power_spectrum.shape)
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

        tbin = np.bincount(r.ravel(), power_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Extracted Noise (Noisy - Clean)")
        plt.imshow(noise_roi, cmap='gray')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("2D Noise Power Spectrum (Log)")
        plt.imshow(np.log1p(power_spectrum), cmap='jet')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("1D Radial NPS")
        plt.plot(radial_profile[:min(h, w) // 2], color='red')
        plt.xlabel("Spatial Frequency")
        plt.ylabel("Power")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def process(self, idx: int = 0):
        if idx < 0 or idx >= len(self.clean_files):
            raise IndexError(f"idx {idx} 超出范围 (0 - {len(self.clean_files) - 1})")

        # 修复了文件读取路径
        file_clean_path = os.path.join(self.clean_img_folder_path, self.clean_files[idx])
        file_noisy_path = os.path.join(self.noisy_img_folder_path, self.noisy_files[idx])
        print(f"🔍 正在分析文件: {self.clean_files[idx]}")

        img_clean = np.load(file_clean_path).squeeze()
        img_noisy = np.load(file_noisy_path).squeeze()

        roi_slice = self.select_roi(img_clean)

        roi_clean = img_clean[roi_slice]
        roi_noisy = img_noisy[roi_slice]

        noise_roi = roi_noisy - roi_clean

        self.statistical_analysis(noise_roi)
        self.power_spectrum_analysis(noise_roi)


if __name__ == "__main__":
    # 注意：确保这里填的是你转换好的 .npy 文件夹路径
    clean = r"G:\PythonProject\MSU_CT_Denoising\notebooks\222\NDCT"
    noisy = r"G:\PythonProject\MSU_CT_Denoising\notebooks\Simulation\Level_0.1_Low-dose_Simulation"

    analysis = NoiseAnalysis(clean_img_folder_path=clean, noisy_img_folder_path=noisy)

    # 开始测试第 2 张图片 (idx=1)
    analysis.process(idx=2)

