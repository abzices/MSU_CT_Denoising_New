import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as vutils

# ✅ TensorBoard
from torch.utils.tensorboard import SummaryWriter

from monai.networks.nets import AutoencoderKL
from dataset import AAPMDenoisingDataset, Config

class AsymmetricFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.5, kl_weight=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kl_weight = kl_weight

    def forward(self, recon, target, z_mu, z_sigma):
        loss_ll = F.l1_loss(recon[:, 0:1, :, :], target[:, 0:1, :, :])
        loss_high = F.l1_loss(recon[:, 1:4, :, :], target[:, 1:4, :, :])
        recon_loss = self.alpha * loss_ll + self.beta * loss_high

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1,2,3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

def train_dwt_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    save_dir = r"D:\wcc\MSU_CT_Denoising\models\monai_vae_checkpoints"
    log_dir = r"D:\wcc\MSU_CT_Denoising\runs\vae_denoising"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "best_dwt_vae.pth")

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    config = Config()
    dataset = AAPMDenoisingDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # 模型
    vae = AutoencoderKL(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=(64, 128, 128, 256),
        latent_channels=8,
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False, True)
    ).to(device)

    # 加载最优模型
    if os.path.exists(best_model_path):
        print(f"✅ 加载预训练模型：{best_model_path}")
        vae.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("❌ 未找到模型，从头开始训练")

    optimizer = optim.AdamW(vae.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = AsymmetricFrequencyLoss(alpha=1.0, beta=2.5, kl_weight=1e-6)

    # AMP
    scaler = torch.amp.GradScaler('cuda')

    epochs = 50
    best_loss = float('inf')
    global_step = 0

    print("🔥 开始训练（AMP + TensorBoard）...")
    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            ldct, ndct, _ = batch
            ldct = ldct.to(device)
            ndct = ndct.to(device)
            x = torch.cat([ndct, ldct], dim=0)

            optimizer.zero_grad()

            # ✅ 新版 AMP 写法（无警告）
            with torch.amp.autocast('cuda'):
                recon, z_mu, z_sigma = vae(x)
                loss, recon_loss, kl_loss = criterion(recon, x, z_mu, z_sigma)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            # TensorBoard 损失
            writer.add_scalar("Train/Total Loss", loss.item(), global_step)
            writer.add_scalar("Train/Recon Loss", recon_loss.item(), global_step)
            writer.add_scalar("Train/KL Loss", kl_loss.item(), global_step)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}"
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} 平均 Loss: {avg_loss:.4f}")

        # 每个 Epoch 画图
        with torch.no_grad():
            vae.eval()
            with torch.amp.autocast('cuda'):
                val_recon, _, _ = vae(x[:4])

            img_gt = x[:4, 0:1, :, :]
            img_recon = val_recon[:4, 0:1, :, :]
            img_comparison = torch.cat([img_gt, img_recon], dim=0)

            writer.add_images(
                "Comparison/GT_vs_Recon",
                img_comparison,
                epoch,
                dataformats="NCHW"
            )

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vae.state_dict(), best_model_path)
            print("✅ 已更新最优模型")

    writer.close()
    print("🏁 训练完成！")

if __name__ == "__main__":
    train_dwt_vae()