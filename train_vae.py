import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# 导入你定义的模型和数据集
from resnet_vae import ResNet50_DWT_VAE
from vgg_vae import VGG19_DWT_VAE
from dataset import AAPMDenoisingDataset, Config as DatasetConfig


# ===================== 1. 训练配置（整合KL退火参数） =====================
@dataclass
class TrainConfig:
    # 基础路径配置
    save_dir: str = "./models"  # 模型保存路径
    tensorboard_dir: str = r"./runs/vgg_vae_pretrain"
    ndct_dir: str = r"./data/Processed/AAPM_Dataset/NDCT"
    ldct_dir: str = r"./data/Processed/AAPM_Dataset/LDCT"
    wavelet_base: str = 'haar'

    # 训练超参
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    num_epochs_stage1: int = 20  # 阶段1：仅训练解码器
    num_epochs_stage2: int = 50  # 阶段2：微调全模型
    lr_stage1: float = 1e-4  # 解码器学习率
    lr_stage2: float = 5e-5  # 全模型微调学习率（更小）
    weight_decay: float = 1e-5  # 权重衰减

    # 损失函数权重（移除固定KL权重，替换为KL退火参数）
    l1_weight: float = 1.0  # L1损失权重
    mse_weight: float = 1.0  # MSE损失权重
    high_freq_weight: float = 2.0  # 高频通道重建损失权重
    kl_anneal_start: int = 0  # KL退火起始步数
    kl_anneal_end: int = 10000  # KL退火结束步数
    kl_weight_max: float = 1e-3  # KL权重最大值

    # 其他
    log_interval: int = 5  # 每多少步打印一次损失
    save_interval: int = 10  # 每多少轮保存一次模型
    val_split: float = 0.1  # 验证集比例


# ===================== 2. KL退火核心函数（迁移自ResNet_VAE.py） =====================
def get_kl_weight(current_step: int, config: TrainConfig) -> float:
    """
    计算动态KL权重（线性退火策略）
    :param current_step: 当前全局训练步数
    :param config: 训练配置
    :return: 动态KL权重
    """
    if current_step < config.kl_anneal_start:
        weight = 0.0
    elif current_step > config.kl_anneal_end:
        weight = config.kl_weight_max
    else:
        # 线性插值计算退火权重
        weight = config.kl_weight_max * (current_step - config.kl_anneal_start) / (
                config.kl_anneal_end - config.kl_anneal_start
        )
    return weight


# ===================== 3. 自定义损失函数（整合动态KL权重） =====================
class VAELoss(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_x, x, mu, logvar, kl_weight: float):
        """
        计算总损失：基础损失(L1+MSE) + 动态KL损失 + 高频通道损失
        :param recon_x: 重建的4通道小波特征 (B,4,H,W)
        :param x: 原始目标4通道小波特征 (B,4,H,W)
        :param mu: 均值 (B,C,H,W)
        :param logvar: 对数方差 (B,C,H,W)
        :param kl_weight: 动态KL权重（由退火策略计算）
        :return: 总损失、各分项损失（用于监控）
        """
        # 1. 基础重建损失（L1 + MSE）
        l1 = self.l1_loss(recon_x, x)
        mse = self.mse_loss(recon_x, x)
        base_loss = self.config.l1_weight * l1 + self.config.mse_weight * mse

        # 2. KL散度损失（VAE标准公式，适配2D特征图）
        # 对所有维度求和（batch, channel, h, w），并归一化batch size
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_div = kl_div / x.size(0)  # 按batch size归一化
        kl_loss = kl_weight * kl_div  # 应用动态KL权重

        # 3. 高频通道重建损失（小波的LH/HL/HH通道，对应索引1/2/3）
        recon_high_freq = recon_x[:, 1:, :, :]
        target_high_freq = x[:, 1:, :, :]
        high_freq_loss = self.l1_loss(recon_high_freq, target_high_freq) * self.config.high_freq_weight

        # 总损失 = 基础损失 + 动态KL损失 + 高频损失
        total_loss = base_loss + kl_loss + high_freq_loss

        return total_loss, {
            "base_loss": base_loss.item(),
            "l1_loss": l1.item(),
            "mse_loss": mse.item(),
            "kl_loss": kl_div.item(),  # 原始KL散度（未乘权重）
            "kl_weight": kl_weight,  # 当前KL权重
            "weighted_kl_loss": kl_loss.item(),  # 加权后的KL损失
            "high_freq_loss": high_freq_loss.item()
        }


# ===================== 4. 数据加载（划分训练/验证集） =====================
def get_dataloaders(config: TrainConfig):
    # 初始化数据集
    dataset_config = DatasetConfig(
        ndct_dir=config.ndct_dir,
        ldct_dir=config.ldct_dir,
        wavelet_base=config.wavelet_base
    )
    full_dataset = AAPMDenoisingDataset(dataset_config)

    # 划分训练/验证集
    val_size = int(config.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定种子，确保划分一致
    )

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True  # num_workers根据CPU核心数调整
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"数据加载完成 | 训练集: {len(train_dataset)} 样本 | 验证集: {len(val_dataset)} 样本")
    return train_loader, val_loader


# ===================== 5. 训练工具函数 =====================
def save_model(model, optimizer, epoch, loss, current_step, save_path):
    """保存模型检查点（新增当前步数，用于恢复训练）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'current_step': current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"模型已保存至: {save_path}")


def validate(model, val_loader, loss_fn, config, current_step):
    """验证模型性能（无梯度计算）"""
    model.eval()
    val_losses = {
        "total_loss": 0.0,
        "base_loss": 0.0,
        "l1_loss": 0.0,
        "mse_loss": 0.0,
        "kl_loss": 0.0,
        "weighted_kl_loss": 0.0,
        "high_freq_loss": 0.0,
        "kl_weight": 0.0
    }

    # 验证阶段使用当前步数的KL权重（或固定最大值，根据需求调整）
    val_kl_weight = get_kl_weight(current_step, config)

    with torch.no_grad():
        for ldct, ndct, _ in tqdm(val_loader, desc="验证中", leave=False):
            ldct = ldct.to(config.device)
            ndct = ndct.to(config.device)

            x = torch.cat([ldct, ndct], dim=0).to(config.device)

            recon, mu, logvar = model(x)
            total_loss, loss_dict = loss_fn(recon, x, mu, logvar, val_kl_weight)

            # 累加损失
            val_losses["total_loss"] += total_loss.item()
            for k in loss_dict.keys():
                val_losses[k] += loss_dict[k]

    # 计算平均损失
    for k in val_losses.keys():
        val_losses[k] /= len(val_loader)

    model.train()
    return val_losses


# ===================== 6. 分阶段训练主逻辑 =====================
def train_stage1(model, train_loader, val_loader, loss_fn, config, writer):
    """阶段1：冻结Encoder，仅训练Decoder"""
    print("\n========== 阶段1：训练解码器 (冻结Encoder) ==========")

    # 冻结Encoder所有层（conv1/layer1/layer2/quant_conv等）
    for name, param in model.named_parameters():
        # Decoder相关参数：quant_conv、decoder
        if "quant_conv" in name or "decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False  # 冻结Encoder

    # 仅优化Decoder参数
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr_stage1,
        weight_decay=config.weight_decay
    )

    # 初始化步数（阶段1和阶段2不共享步数计数）
    current_step = 0

    # 训练循环
    model.train()
    for epoch in range(config.num_epochs_stage1):
        train_losses = {
            "total_loss": 0.0,
            "base_loss": 0.0,
            "l1_loss": 0.0,
            "mse_loss": 0.0,
            "kl_loss": 0.0,
            "weighted_kl_loss": 0.0,
            "high_freq_loss": 0.0,
            "kl_weight": 0.0
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs_stage1}")
        for batch_idx, (ldct, ndct, _) in enumerate(pbar):
            # 计算当前KL权重
            kl_weight = get_kl_weight(current_step, config)

            # 数据移至设备
            ldct = ldct.to(config.device)
            ndct = ndct.to(config.device)
            x = torch.cat([ldct, ndct], dim=0).to(config.device)

            # 前向传播
            recon, mu, logvar = model(x)
            total_loss, loss_dict = loss_fn(recon, x, mu, logvar, kl_weight)

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 累加训练损失
            train_losses["total_loss"] += total_loss.item()
            for k in loss_dict.keys():
                train_losses[k] += loss_dict[k]

            # 全局步数自增
            current_step += 1

            # 打印平滑的训练日志
            if (batch_idx + 1) % config.log_interval == 0:
                current_steps = batch_idx + 1
                pbar.set_postfix({
                    "Total": f"{train_losses['total_loss'] / current_steps:.4f}",
                    "Base(L1)": f"{train_losses['base_loss'] / current_steps:.4f}",
                    "HighFreq": f"{train_losses['high_freq_loss'] / current_steps:.4f}",
                    "KL(weight)": f"{loss_dict['kl_weight']:.4f}",
                    "KL(loss)": f"{loss_dict['weighted_kl_loss']:.6f}"
                })

                # 记录训练阶段的 Loss 到 TensorBoard
                writer.add_scalar('Stage1/Train/Total_Loss', train_losses['total_loss'] / current_steps, current_step)
                writer.add_scalar('Stage1/Train/Base_Loss', train_losses['base_loss'] / current_steps, current_step)
                writer.add_scalar('Stage1/Train/HighFreq_Loss', train_losses['high_freq_loss'] / current_steps,
                                  current_step)
                writer.add_scalar('Stage1/Train/KL_Loss_Weighted', loss_dict['weighted_kl_loss'], current_step)
                writer.add_scalar('Stage1/Train/KL_Weight', loss_dict['kl_weight'], current_step)

        # 验证集评估
        val_losses = validate(model, val_loader, loss_fn, config, current_step)
        print(f"\nEpoch {epoch + 1} 验证集损失：")
        print(f"总损失: {val_losses['total_loss']:.4f} | 基础损失: {val_losses['base_loss']:.4f} | "
              f"高频损失: {val_losses['high_freq_loss']:.4f} | 原始KL损失: {val_losses['kl_loss']:.6f} | "
              f"KL权重: {val_losses['kl_weight']:.4f} | 加权KL损失: {val_losses['weighted_kl_loss']:.6f}")
        # 记录验证集 Loss 到 TensorBoard
        writer.add_scalar('Stage1/Val/Total_Loss', val_losses['total_loss'], current_step)
        writer.add_scalar('Stage1/Val/HighFreq_Loss', val_losses['high_freq_loss'], current_step)

        # 每轮验证后，抽几张图的低频通道(LL)放进 TensorBoard 看效果
        with torch.no_grad():
            # 从当前的 x (batch中的最后几张图) 和 recon 提取第 0 个通道 (低频大体结构)
            # 取前 4 张图，便于拼图展示
            img_target = x[:4, 0:1, :, :]
            img_recon = recon[:4, 0:1, :, :]
            # 将原图和重建图上下拼接
            img_grid = vutils.make_grid(torch.cat([img_target, img_recon], dim=0), nrow=4, normalize=True,
                                        scale_each=True)
            writer.add_image('Stage1/Reconstruction_Images', img_grid, current_step)

        # 保存模型
        if (epoch + 1) % config.save_interval == 0:
            save_path = os.path.join(config.save_dir, f"stage1_epoch_{epoch + 1}.pth")
            save_model(model, optimizer, epoch + 1, val_losses["total_loss"], current_step, save_path)

    # 保存阶段1最终模型
    final_save_path = os.path.join(config.save_dir, "stage1_final.pth")
    save_model(model, optimizer, config.num_epochs_stage1, val_losses["total_loss"], current_step, final_save_path)
    print("========== 阶段1训练完成 ==========\n")
    return current_step  # 返回当前步数，供阶段2继续使用


def train_stage2(model, train_loader, val_loader, loss_fn, config, start_step, writer):
    """阶段2：解冻所有层，微调全模型"""
    print("\n========== 阶段2：微调全模型 (解冻Encoder) ==========")

    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    # 优化器（学习率更小，可加学习率调度）
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr_stage2,
        weight_decay=config.weight_decay
    )
    # 学习率调度器（按需启用）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 不继承阶段1的全局步数
    current_step = 0

    # 训练循环
    model.train()
    for epoch in range(config.num_epochs_stage2):
        train_losses = {
            "total_loss": 0.0,
            "base_loss": 0.0,
            "l1_loss": 0.0,
            "mse_loss": 0.0,
            "kl_loss": 0.0,
            "weighted_kl_loss": 0.0,
            "high_freq_loss": 0.0,
            "kl_weight": 0.0
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs_stage2}")
        for batch_idx, (ldct, ndct, _) in enumerate(pbar):
            # 计算当前KL权重
            kl_weight = get_kl_weight(current_step, config)

            # 数据移至设备
            ldct = ldct.to(config.device)
            ndct = ndct.to(config.device)
            x = torch.cat([ldct, ndct], dim=0).to(config.device)

            # 前向传播
            recon, mu, logvar = model(x)
            total_loss, loss_dict = loss_fn(recon, x, mu, logvar, kl_weight)

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 累加损失
            train_losses["total_loss"] += total_loss.item()
            for k in loss_dict.keys():
                train_losses[k] += loss_dict[k]

            # 全局步数自增
            current_step += 1

            # 打印平滑的训练日志
            if (batch_idx + 1) % config.log_interval == 0:
                current_steps = batch_idx + 1
                pbar.set_postfix({
                    "Total": f"{train_losses['total_loss'] / current_steps:.4f}",
                    "Base(L1)": f"{train_losses['base_loss'] / current_steps:.4f}",
                    "HighFreq": f"{train_losses['high_freq_loss'] / current_steps:.4f}",
                    "KL(weight)": f"{loss_dict['kl_weight']:.4f}",
                    "KL(loss)": f"{loss_dict['weighted_kl_loss']:.6f}"
                })
                # 👈 新增：记录训练阶段的 Loss 到 TensorBoard
                writer.add_scalar('Stage2/Train/Total_Loss', train_losses['total_loss'] / current_steps, current_step)
                writer.add_scalar('Stage2/Train/Base_Loss', train_losses['base_loss'] / current_steps, current_step)
                writer.add_scalar('Stage2/Train/HighFreq_Loss', train_losses['high_freq_loss'] / current_steps,
                                  current_step)
                writer.add_scalar('Stage2/Train/KL_Loss_Weighted', loss_dict['weighted_kl_loss'], current_step)
                writer.add_scalar('Stage2/Train/KL_Weight', loss_dict['kl_weight'], current_step)

        # 验证集评估
        val_losses = validate(model, val_loader, loss_fn, config, current_step)
        print(f"\nEpoch {epoch + 1} 验证集损失：")
        print(f"总损失: {val_losses['total_loss']:.4f} | 基础损失: {val_losses['base_loss']:.4f} | "
              f"高频损失: {val_losses['high_freq_loss']:.4f} | 原始KL损失: {val_losses['kl_loss']:.6f} | "
              f"KL权重: {val_losses['kl_weight']:.4f} | 加权KL损失: {val_losses['weighted_kl_loss']:.6f}")

        # 记录验证集 Loss 到 TensorBoard
        writer.add_scalar('Stage2/Val/Total_Loss', val_losses['total_loss'], current_step)
        writer.add_scalar('Stage2/Val/HighFreq_Loss', val_losses['high_freq_loss'], current_step)

        # 每轮验证后，抽几张图的低频通道(LL)放进 TensorBoard 看效果
        with torch.no_grad():
            # 从当前的 x (batch中的最后几张图) 和 recon 提取第 0 个通道 (低频大体结构)
            # 取前 4 张图，便于拼图展示
            img_target = x[:4, 0:1, :, :]
            img_recon = recon[:4, 0:1, :, :]
            # 将原图和重建图上下拼接
            img_grid = vutils.make_grid(torch.cat([img_target, img_recon], dim=0), nrow=4, normalize=True,
                                        scale_each=True)
            writer.add_image('Stage2/Reconstruction_Images', img_grid, current_step)

        # 学习率调度
        scheduler.step(val_losses["total_loss"])

        # 保存模型
        if (epoch + 1) % config.save_interval == 0:
            save_path = os.path.join(config.save_dir, f"stage2_epoch_{epoch + 1}.pth")
            save_model(model, optimizer, epoch + 1, val_losses["total_loss"], current_step, save_path)

    # 保存阶段2最终模型
    final_save_path = os.path.join(config.save_dir, "stage2_final.pth")
    save_model(model, optimizer, config.num_epochs_stage2, val_losses["total_loss"], current_step, final_save_path)
    print("========== 阶段2训练完成 ==========\n")


# ===================== 7. 主函数（一键运行） =====================
if __name__ == "__main__":
    # 初始化配置
    config = TrainConfig()
    print(f"训练配置 | 设备: {config.device} | 批次大小: {config.batch_size} | "
          f"阶段1轮数: {config.num_epochs_stage1} | 阶段2轮数: {config.num_epochs_stage2} | "
          f"KL退火区间: [{config.kl_anneal_start}, {config.kl_anneal_end}] | 最大KL权重: {config.kl_weight_max}")

    # 加载数据
    train_loader, val_loader = get_dataloaders(config)

    # 初始化模型
    model = VGG19_DWT_VAE(
        latent_channels=8
    ).to(config.device)
    print(f"模型初始化完成 | 设备: {next(model.parameters()).device}")

    # 初始化损失函数
    loss_fn = VAELoss(config).to(config.device)

    # 初始化 TensorBoard Writer
    writer = SummaryWriter(log_dir=config.tensorboard_dir)
    print(f"TensorBoard 日志已保存至: {config.tensorboard_dir}")

    # 阶段1：训练解码器
    #stage1_end_step = train_stage1(model, train_loader, val_loader, loss_fn, config, writer)

    # 直接开始阶段2的训练
    checkpoint = torch.load(os.path.join(config.save_dir, "stage1_epoch_10.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    stage1_end_step = checkpoint['current_step']
    print("已成功加载 Stage 1 预训练权重！准备进入 Stage 2...")

    # 阶段2：微调全模型（传入阶段1结束的步数）
    train_stage2(model, train_loader, val_loader, loss_fn, config, stage1_end_step, writer)

    # 训练结束后关闭 writer
    writer.close()
    print("所有训练阶段完成！最终模型已保存至:", config.save_dir)