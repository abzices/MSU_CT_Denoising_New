import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

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

    save_dir = r"G:\PythonProject\MSU_CT_Denoising\models\vae_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    config = Config()
    dataset = AAPMDenoisingDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # ✅ 修复后的 AutoencoderKL
    vae = AutoencoderKL(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=(64, 128, 128, 256),  # ✅ 这里修复了
        latent_channels=8,
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False, True)
    ).to(device)

    optimizer = optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = AsymmetricFrequencyLoss(alpha=1.0, beta=2.5, kl_weight=1e-6)

    epochs = 50
    best_loss = float('inf')

    print("🔥 开始训练频域 VAE...")
    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            # dataset返回: (ldct_tensor, ndct_tensor, filename)
            ldct, ndct, _ = batch
            # 将数据移至设备
            ldct = ldct.to(device)
            ndct = ndct.to(device)
            # 拼接NDCT和LDCT作为输入
            x = torch.cat([ndct, ldct], dim=0)

            optimizer.zero_grad()
            # VAE前向传播
            recon, z_mu, z_sigma = vae(x)
            # 计算损失
            loss, recon_loss, kl_loss = criterion(recon, x, z_mu, z_sigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Recon": f"{recon_loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} 平均 Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vae.state_dict(), os.path.join(save_dir, "best_dwt_vae.pth"))
            print("✅ 已保存最优模型")

if __name__ == "__main__":
    train_dwt_vae()