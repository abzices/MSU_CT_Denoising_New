"""
ResNet VAE 模型架构可视化脚本
"""
import torch
from src.models.resnet_fromzero import ResNet50Autoencoder, ResNet18Autoencoder


def print_model_info(model, model_name):
    """打印模型详细信息"""
    print("=" * 80)
    print(f"{model_name} 模型信息")
    print("=" * 80)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数量(MB): {total_params * 4 / 1024 / 1024:.2f}")  # 假设float32
    
    # 测试前向传播
    dummy_input = torch.randn(1, 4, 256, 256)
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(dummy_input)
    
    print(f"\n输入尺寸: {dummy_input.shape}")
    print(f"潜在空间均值尺寸: {mu.shape}")
    print(f"潜在空间对数方差尺寸: {logvar.shape}")
    print(f"重建输出尺寸: {recon.shape}")
    
    # 打印编码器结构
    print(f"\n编码器结构:")
    print(f"  - conv1: 4 -> 64, 7x7, stride=2")
    print(f"  - maxpool: 3x3, stride=2")
    print(f"  - layer1: 64 -> 256 (ResNet50) / 64 (ResNet18)")
    print(f"  - layer2: 256 -> 512 (ResNet50) / 64 -> 128 (ResNet18)")
    print(f"  - layer3: 512 -> 1024 (ResNet50) / 128 -> 256 (ResNet18)")
    print(f"  - layer4: 1024 -> 2048 (ResNet50) / 256 -> 512 (ResNet18)")
    print(f"  - quant_conv_mu: 2048 -> 8 (ResNet50) / 512 -> 8 (ResNet18)")
    print(f"  - quant_conv_logvar: 2048 -> 8 (ResNet50) / 512 -> 8 (ResNet18)")
    
    # 打印解码器结构
    print(f"\n解码器结构:")
    print(f"  - post_quant_conv: 8 -> 2048 (ResNet50) / 8 -> 512 (ResNet18)")
    print(f"  - deconv4: 2048 -> 1024 (ResNet50) / 512 -> 256 (ResNet18)")
    print(f"  - deconv3: 1024 -> 512 (ResNet50) / 256 -> 128 (ResNet18)")
    print(f"  - deconv2: 512 -> 256 (ResNet50) / 128 -> 64 (ResNet18)")
    print(f"  - deconv1: 256 -> 64 (ResNet50) / 64 -> 64 (ResNet18)")
    print(f"  - final_conv: 64 -> 32 -> 4")
    
    print("\n" + "=" * 80)


def print_layer_shapes(model, model_name):
    """打印每一层的输出形状"""
    print(f"\n{model_name} 各层输出形状:")
    print("=" * 80)
    
    dummy_input = torch.randn(1, 4, 256, 256)
    model.eval()
    
    # 编码器
    print("\n编码器:")
    x = dummy_input
    print(f"输入: {x.shape}")
    
    x = model.encoder.conv1(x)
    print(f"conv1: {x.shape}")
    
    x = model.encoder.bn1(x)
    x = model.encoder.relu(x)
    x = model.encoder.maxpool(x)
    print(f"maxpool: {x.shape}")
    
    x = model.encoder.layer1(x)
    print(f"layer1: {x.shape}")
    
    x = model.encoder.layer2(x)
    print(f"layer2: {x.shape}")
    
    x = model.encoder.layer3(x)
    print(f"layer3: {x.shape}")
    
    x = model.encoder.layer4(x)
    print(f"layer4: {x.shape}")
    
    mu = model.encoder.quant_conv_mu(x)
    logvar = model.encoder.quant_conv_logvar(x)
    print(f"quant_conv_mu: {mu.shape}")
    print(f"quant_conv_logvar: {logvar.shape}")
    
    # 解码器
    print("\n解码器:")
    z = mu  # 使用均值作为潜在表示
    print(f"潜在空间输入: {z.shape}")
    
    x = model.decoder.post_quant_conv(z)
    print(f"post_quant_conv: {x.shape}")
    
    x = model.decoder.deconv4(x)
    print(f"deconv4: {x.shape}")
    
    x = model.decoder.deconv3(x)
    print(f"deconv3: {x.shape}")
    
    x = model.decoder.deconv2(x)
    print(f"deconv2: {x.shape}")
    
    x = model.decoder.deconv1(x)
    print(f"deconv1: {x.shape}")
    
    x = model.decoder.final_conv(x)
    print(f"final_conv: {x.shape}")
    
    print("\n" + "=" * 80)


def compare_models():
    """比较两个模型的性能"""
    print("\n模型对比:")
    print("=" * 80)
    
    # 创建模型
    model50 = ResNet50Autoencoder(latent_channels=8)
    model18 = ResNet18Autoencoder(latent_channels=8)
    
    # 计算参数量
    params50 = sum(p.numel() for p in model50.parameters())
    params18 = sum(p.numel() for p in model18.parameters())
    
    print(f"{'指标':<20} {'ResNet50':<15} {'ResNet18':<15} {'差异':<15}")
    print("-" * 80)
    print(f"{'参数量':<20} {params50:>14,} {params18:>14,} {params50-params18:>14,}")
    print(f"{'参数量(MB)':<20} {params50*4/1024/1024:>14.2f} {params18*4/1024/1024:>14.2f} {(params50-params18)*4/1024/1024:>14.2f}")
    print(f"{'压缩比':<20} {params50/params18:>14.2f}x {1:>14.2f}x {params50/params18:>14.2f}x")
    
    # 测试推理速度
    import time
    
    dummy_input = torch.randn(1, 4, 256, 256)
    
    model50.eval()
    model18.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model50(dummy_input)
            _ = model18(dummy_input)
    
    # 测试ResNet50
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model50(dummy_input)
    time50 = time.time() - start_time
    
    # 测试ResNet18
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model18(dummy_input)
    time18 = time.time() - start_time
    
    print(f"{'推理时间(100次)':<20} {time50:>14.4f}s {time18:>14.4f}s {time50-time18:>14.4f}s")
    print(f"{'平均推理时间':<20} {time50/100:>14.6f}s {time18/100:>14.6f}s {(time50-time18)/100:>14.6f}s")
    print(f"{'速度比':<20} {1:>14.2f}x {time50/time18:>14.2f}x {time50/time18:>14.2f}x")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ResNet VAE 模型架构可视化")
    print("=" * 80)
    
    # 创建模型
    model50 = ResNet50Autoencoder(latent_channels=8)
    model18 = ResNet18Autoencoder(latent_channels=8)
    
    # 打印模型信息
    print_model_info(model50, "ResNet50Autoencoder")
    print_model_info(model18, "ResNet18Autoencoder")
    
    # 打印层形状
    print_layer_shapes(model50, "ResNet50Autoencoder")
    print_layer_shapes(model18, "ResNet18Autoencoder")
    
    # 比较模型
    compare_models()
    
    print("\n可视化完成！")
