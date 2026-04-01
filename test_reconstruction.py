import torch
import pywt
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import AutoencoderKL

# 1. 初始化相同的模型并加载权重
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoencoderKL(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=(64, 128, 128, 256),
        latent_channels=8,
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False, True)
    ).to(device)

# 💡 替换为你第 30 个 Epoch 的模型路径！
checkpoint = torch.load("./models/monai_vae/best_dwt_vae.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 2. 随便拿一张验证集里的离线 DWT Tensor (.pt)
# 💡 替换为你实际的 .pt 文件路径！
test_pt_path = r"./data/Processed/AAPM_Dataset/NDCT_DWT_PT/L067_FD_1_SHARP_1.CT.0002.0021.2016.01.21.18.11.40.977560.404629495.pt"
x = torch.load(test_pt_path, weights_only=True).unsqueeze(0).to(device) # shape:[1, 4, 256, 256]

# 3. 让 VAE 进行重建
with torch.no_grad():
    recon, _, _ = model(x)

# 4. 把 Tensor 转换回 Numpy，并乘回 2.0 (解掉之前的归一化)
x_np = x.squeeze().cpu().numpy() * 2.0
recon_np = recon.squeeze().cpu().numpy() * 2.0

# 5. 💡 逆小波变换 (IDWT) 拼回 512x512！
def idwt_reconstruct(dwt_array):
    cA = dwt_array[0]
    cH = dwt_array[1]
    cV = dwt_array[2]
    cD = dwt_array[3]
    # 使用 haar 小波重构
    recon_img = pywt.idwt2((cA, (cH, cV, cD)), 'haar', mode='symmetric')
    return recon_img

original_512 = idwt_reconstruct(x_np)
reconstructed_512 = idwt_reconstruct(recon_np)

# 6. 画图对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# 设置统一的灰度窗宽窗位 (由于之前归一化在[-1,1]，对应真实HU是 -1000~400)
vmin, vmax = -1.0, 1.0

axes[0].imshow(original_512, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title("Original 512x512 CT")
axes[0].axis('off')

axes[1].imshow(reconstructed_512, cmap='gray', vmin=vmin, vmax=vmax)
axes[1].set_title(f"VAE Reconstructed (Loss: ~0.13)")
axes[1].axis('off')

# 画出误差图（乘以倍数放大误差）
error_map = np.abs(original_512 - reconstructed_512)
im = axes[2].imshow(error_map, cmap='hot', vmin=0, vmax=0.2)
axes[2].set_title("Absolute Error Map")
axes[2].axis('off')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()