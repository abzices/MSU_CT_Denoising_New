import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from tqdm import tqdm
import torchvision.utils as vutils
import torchvision.models as models

# ✅ TensorBoard 与 MONAI
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import AutoencoderKL


# ==========================================
# 1. 全局训练配置 (Hyperparameters)
# ==========================================
@dataclass
class TrainConfig:
    # 💡 请确保这里的路径指向你已经提前用 pywt 提取好的 .pt 张量文件夹！
    ndct_dir: str = r"./data/Processed/AAPM_Dataset/NDCT_DWT_PT"
    ldct_dir: str = r"./data/Processed/AAPM_Dataset/LDCT_DWT_PT"

    save_dir: str = r"./models/monai_vae_checkpoints"
    log_dir: str = r"./runs/vae_denoising_perceptual"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8  # 如果加了VGG显存不够，可以降到 4
    num_epochs: int = 50
    learning_rate: float = 3e-4  # AdamW 的黄金起步学习率
    weight_decay: float = 1e-5

    # 💡 损失函数权重（融合了所有我们在学术上探讨过的最优比例）
    l1_weight: float = 1.0  # 低频大轮廓L1
    mse_weight: float = 1.0  # 低频大轮廓MSE
    high_freq_weight: float = 1.0  # 高频病灶惩罚
    perceptual_weight: float = 0.05  # 💡 VGG感知损失权重（治愈过度平滑的终极武器）

    # KL 退火参数
    kl_anneal_start: int = 0
    kl_anneal_end: int = 5000
    kl_weight_max: float = 1e-3  # 微弱的KL散度约束即可

    val_split: float = 0.05

    # 其他
    log_interval: int = 5  # 每多少步打印一次损失
    save_interval: int = 10  # 每多少轮保存一次模型


# ==========================================
# 2. 极速数据集 (直接读取 .pt)
# ==========================================
class AAPMDenoisingPTDataset(Dataset):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.ndct_paths = sorted(glob.glob(os.path.join(config.ndct_dir, "*.pt")))
        self.ldct_paths = sorted(glob.glob(os.path.join(config.ldct_dir, "*.pt")))
        assert len(self.ndct_paths) == len(self.ldct_paths) and len(self.ndct_paths) > 0, "数据配对失败或未找到.pt文件"

    def __len__(self):
        return len(self.ndct_paths)

    def __getitem__(self, idx):
        # 极速 I/O，无需 CPU 参与计算
        ndct_tensor = torch.load(self.ndct_paths[idx], weights_only=True)
        ldct_tensor = torch.load(self.ldct_paths[idx], weights_only=True)
        return ldct_tensor, ndct_tensor


# ==========================================
# 3. 损失函数体系 (感知损失 + 高低频不对称 + KL)
# ==========================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights="IMAGENET1K_V1").features
        self.blocks = nn.ModuleList([
            vgg[:4],  # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16],  # relu3_3
            vgg[16:23]  # relu4_3
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_img, target_img):
        # 提取 LL 通道 (大体轮廓)，伪装成 RGB 输入 VGG
        input_rgb = input_img[:, 0:1, :, :].repeat(1, 3, 1, 1)
        target_rgb = target_img[:, 0:1, :, :].repeat(1, 3, 1, 1)

        # 将 [-1, 1] 变换为 ImageNet 要求的分布
        input_rgb = (input_rgb + 1.0) / 2.0
        target_rgb = (target_rgb + 1.0) / 2.0
        input_rgb = (input_rgb - self.mean) / self.std
        target_rgb = (target_rgb - self.mean) / self.std

        loss = 0.0
        x, y = input_rgb, target_rgb
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


class VAELoss(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss()  # 💡 真正挂载裁判

    def forward(self, recon, target, z_mu, z_sigma, kl_weight):
        # 1. 基础重建 (低频大轮廓 LL)
        l1 = self.l1_loss(recon[:, 0:1], target[:, 0:1])
        mse = self.mse_loss(recon[:, 0:1], target[:, 0:1])

        # 2. 高频不对称惩罚 (LH, HL, HH)
        high_freq = self.l1_loss(recon[:, 1:4], target[:, 1:4])

        # 3. 💡 VGG 感知损失 (强迫生成针状逼真纹理)
        p_loss = self.perceptual_loss(recon, target)

        # 4. KL 散度 (MONAI 输出的是 sigma)
        kl_div = -0.5 * torch.sum(1 + torch.log(z_sigma.pow(2) + 1e-8) - z_mu.pow(2) - z_sigma.pow(2))
        kl_div = kl_div / target.size(0)

        # 加权求和
        total_loss = (self.config.l1_weight * l1) + \
                     (self.config.mse_weight * mse) + \
                     (self.config.high_freq_weight * high_freq) + \
                     (self.config.perceptual_weight * p_loss) + \
                     (kl_weight * kl_div)

        return total_loss, {
            "base": (l1 + mse).item(),
            "high_freq": high_freq.item(),
            "perceptual": p_loss.item(),
            "kl": kl_div.item()
        }


# ==========================================
# 4. 主训练流水线
# ==========================================
def train():
    config = TrainConfig()
    print(f"🔥 设备: {config.device} | Batch: {config.batch_size} | AMP: 开启")

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)

    # 数据准备
    full_dataset = AAPMDenoisingPTDataset(config)
    val_size = int(config.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 模型初始化：强力 MONAI 架构
    vae = AutoencoderKL(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=(64, 128, 256),  # 容量足够消除马赛克
        latent_channels=8,
        num_res_blocks=2,  # 残差块保平滑
        attention_levels=(False, False, True),
        norm_num_groups=32
    ).to(config.device)

    # 可以自动接着之前的权重练
    best_model_path = os.path.join(config.save_dir, "best_dwt_vae.pth")
    if os.path.exists(best_model_path):
        print(f"✅ 加载预训练模型：{best_model_path}")
        vae.load_state_dict(torch.load(best_model_path, map_location=config.device, weights_only=True))

    optimizer = optim.AdamW(vae.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = VAELoss(config).to(config.device)
    scaler = torch.amp.GradScaler('cuda')

    global_step = 0
    best_loss = float('inf')

    for epoch in range(config.num_epochs):
        vae.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for ldct, ndct in pbar:
            ldct, ndct = ldct.to(config.device), ndct.to(config.device)
            # 自重建策略：干净和脏特征一起练
            x = torch.cat([ndct, ldct], dim=0)

            # 计算当前退火 KL 权重
            kl_weight = config.kl_weight_max if global_step > config.kl_anneal_end else \
                (config.kl_weight_max * global_step / config.kl_anneal_end)

            optimizer.zero_grad()

            # 💡 AMP 混合精度前向传播
            with torch.amp.autocast('cuda'):
                recon, z_mu, z_sigma = vae(x)
                loss, loss_dict = criterion(recon, x, z_mu, z_sigma, kl_weight)

            # 💡 梯度缩放与反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            # 日志写入与打印
            if global_step % config.log_interval == 0:
                writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
                writer.add_scalar("Train/Perceptual", loss_dict["perceptual"], global_step)
                writer.add_scalar("Train/KL_Raw", loss_dict["kl"], global_step)

                pbar.set_postfix({
                    "Total": f"{loss.item():.3f}",
                    "Perc": f"{loss_dict['perceptual']:.3f}",
                    "HFreq": f"{loss_dict['high_freq']:.3f}"
                })

        print(f"Epoch {epoch + 1} 平均 Loss: {epoch_loss / len(train_loader):.4f}")

        # ==========================================
        # 验证与出图环节
        # ==========================================
        with torch.no_grad():
            vae.eval()
            val_loss_total = 0.0

            for ldct, ndct in val_loader:
                x_val = torch.cat([ndct.to(config.device), ldct.to(config.device)], dim=0)
                with torch.amp.autocast('cuda'):
                    val_recon, z_mu_v, z_sigma_v = vae(x_val)
                    v_loss, _ = criterion(val_recon, x_val, z_mu_v, z_sigma_v, config.kl_weight_max)
                    val_loss_total += v_loss.item()

            avg_val_loss = val_loss_total / len(val_loader)
            writer.add_scalar("Val/Total_Loss", avg_val_loss, epoch)

            # 画图：提取前4张图的低频通道对比 (原图 vs 重建)
            img_gt = x_val[:4, 0:1, :, :]
            img_rc = val_recon[:4, 0:1, :, :]
            img_grid = vutils.make_grid(torch.cat([img_gt, img_rc], dim=0), nrow=4, normalize=True, scale_each=True)
            writer.add_image("Validation/Original_vs_Recon", img_grid, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(vae.state_dict(), best_model_path)
                print(f"✅ 验证集突破新低: {best_loss:.4f}，模型已保存！")

    writer.close()
    print("🏁 终极 VAE 训练完成！")


if __name__ == "__main__":
    train()