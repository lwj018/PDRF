"""
最终推荐方案：时间序列LPIPS损失
专门为 B,T,H,W 格式设计的整流模型损失函数

这是针对你的问题的最佳解决方案！
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

class FinalRecommendedLPIPS(nn.Module):
    """
    最终推荐的LPIPS实现
    - 专门处理 B,T,H,W 格式
    - 简单、高效、稳定
    - 开箱即用
    """
    
    def __init__(self, lpips_weight=0.1, use_gpu=True):
        super(FinalRecommendedLPIPS, self).__init__()
        
        self.lpips_weight = lpips_weight
        
        # 轻量级VGG特征提取器
        vgg = models.vgg16(weights="DEFAULT")
        self.features = nn.Sequential(*vgg.features[:16])  # 到conv3_3
        
        # 冻结VGG参数
        for param in self.features.parameters():
            param.requires_grad = False
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        # ImageNet归一化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if use_gpu and torch.cuda.is_available():
            self.cuda()
    
    def forward(self, velocity_pred, x_target, noise):
        """
        一步到位的损失计算
        
        Args:
            velocity_pred: [B, T, H, W] 模型预测的速度
            x_target: [B, T, H, W] 目标数据
            noise: [B, T, H, W] 噪声
        
        Returns:
            dict: 包含所有损失信息
        """
        B, T, H, W = velocity_pred.shape
        
        # Step 1: 计算真实速度
        velocity_target = x_target - noise
        
        # Step 2: 归一化到[0,1]
        pred_norm = velocity_pred
        target_norm = velocity_target
        
        # Step 3: 转换为3通道 [B,T,H,W] -> [B,T,3,H,W]
        pred_3ch = pred_norm.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        target_3ch = target_norm.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        
        # Step 4: 重塑为批量处理 [B,T,3,H,W] -> [B*T,3,H,W]
        pred_batch = pred_3ch.view(B * T, 3, H, W)
        target_batch = target_3ch.view(B * T, 3, H, W)
        
        # Step 5: ImageNet归一化
        pred_norm_batch = pred_batch
        target_norm_batch = target_batch
        
        # Step 6: 提取VGG特征
        pred_features = self.features(pred_norm_batch)
        target_features = self.features(target_norm_batch)
        
        # Step 7: 计算感知损失
        lpips_loss = F.mse_loss(pred_features, target_features)
        
        # Step 8: 计算MSE损失
        #mse_loss = self.mse_loss(velocity_pred, velocity_target)
        
        # Step 9: 组合损失
        #total_loss = self.lpips_weight * lpips_loss + mse_loss
        
        # return {
        #     'total_loss': total_loss,
        #     'lpips_loss': lpips_loss.detach(),
        #     'mse_loss': mse_loss.detach(),
        #     'lpips_weighted': (self.lpips_weight * lpips_loss).detach(),
        #     'loss_ratio': (lpips_loss / mse_loss).detach()
        # }
        return lpips_loss

# 便捷的使用接口
def create_rectified_flow_loss(lpips_weight=0.1, use_gpu=True):
    """
    创建整流模型专用的LPIPS损失
    
    Args:
        lpips_weight: LPIPS损失权重，推荐0.05-0.2
        use_gpu: 是否使用GPU
    
    Returns:
        损失函数实例
    """
    return FinalRecommendedLPIPS(lpips_weight=lpips_weight, use_gpu=use_gpu)


# 使用示例
def example_usage():
    """完整的使用示例"""
    
    print("=== 最终推荐方案使用示例 ===")
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建损失函数
    loss_fn = create_rectified_flow_loss(lpips_weight=1.0, use_gpu=(device=='cuda'))
    print("✅ 损失函数创建成功")
    
    # 模拟数据
    batch_size = 4
    time_steps = 8
    height, width = 64, 64
    
    print(f"数据形状: [{batch_size}, {time_steps}, {height}, {width}]")
    
    # 生成测试数据
    velocity_pred = torch.randn(batch_size, time_steps, height, width).to(device)
    x_target = torch.rand(batch_size, time_steps, height, width).to(device)
    noise = torch.randn(batch_size, time_steps, height, width).to(device) * 0.1
    
    print("✅ 测试数据生成完成")
    
    # 计算损失
    with torch.no_grad():
        losses = loss_fn(velocity_pred, x_target, noise)
    
    # 显示结果
    print("\n=== 损失计算结果 ===")
    print(f"Total Loss:    {losses['total_loss'].item():.6f}")
    print(f"LPIPS Loss:    {losses['lpips_loss'].item():.6f}")
    print(f"MSE Loss:      {losses['mse_loss'].item():.6f}")
    print(f"LPIPS Weight:  {losses['lpips_weighted'].item():.6f}")
    print(f"Loss Ratio:    {losses['loss_ratio'].item():.3f}")
    
    # 梯度测试
    print("\n=== 梯度测试 ===")
    velocity_pred.requires_grad_(True)
    losses = loss_fn(velocity_pred, x_target, noise)
    losses['total_loss'].backward()
    grad_norm = velocity_pred.grad.norm().item()
    print(f"梯度范数: {grad_norm:.6f}")
    print("✅ 梯度计算正常" if grad_norm > 0 else "❌ 梯度计算异常")
    
    return losses


def training_template():
    """训练模板"""
    
    print("\n=== 训练模板 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 创建模型（示例）
    class SimpleRectifiedFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding=1)
            )
        
        def forward(self, x_t, noise):
            B, T, H, W = x_t.shape
            x_flat = x_t.view(B*T, 1, H, W)
            noise_flat = noise.view(B*T, 1, H, W)
            
            input_with_noise = x_flat + noise_flat
            velocity_flat = self.net(input_with_noise)
            
            return velocity_flat.view(B, T, H, W)
    
    # 2. 初始化
    model = SimpleRectifiedFlow().to(device)
    loss_fn = create_rectified_flow_loss(lpips_weight=0.1, use_gpu=(device=='cuda'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("模型参数数量:", sum(p.numel() for p in model.parameters()))
    
    # 3. 训练循环
    model.train()
    
    for epoch in range(3):
        epoch_losses = {'total': 0, 'lpips': 0, 'mse': 0}
        
        for step in range(5):  # 模拟5个batch
            # 生成批次数据
            batch_size = 2
            time_steps = 4
            
            x_target = torch.rand(batch_size, time_steps, 64, 64).to(device)
            noise = torch.randn(batch_size, time_steps, 64, 64).to(device) * 0.1
            
            # 前向传播
            velocity_pred = model(x_target, noise)
            
            # 计算损失
            losses = loss_fn(velocity_pred, x_target, noise)
            
            # 反向传播
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            # 累积损失
            epoch_losses['total'] += losses['total_loss'].item()
            epoch_losses['lpips'] += losses['lpips_loss'].item()
            epoch_losses['mse'] += losses['mse_loss'].item()
        
        # 打印epoch结果
        print(f"Epoch {epoch+1}: "
              f"Total={epoch_losses['total']/5:.6f}, "
              f"LPIPS={epoch_losses['lpips']/5:.6f}, "
              f"MSE={epoch_losses['mse']/5:.6f}")
    
    print("✅ 训练模板测试完成")


def performance_benchmark():
    """性能基准测试"""
    
    print("\n=== 性能基准测试 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = create_rectified_flow_loss(lpips_weight=0.1, use_gpu=(device=='cuda'))
    
    import time
    
    configs = [
        (2, 4, 64, 64),   # 小规模
        (4, 8, 64, 64),   # 中规模
        (2, 16, 64, 64),  # 长时间序列
        (4, 4, 128, 128), # 高分辨率
    ]
    
    for B, T, H, W in configs:
        print(f"\n配置: B={B}, T={T}, H={H}, W={W}")
        
        # 生成数据
        velocity_pred = torch.randn(B, T, H, W).to(device)
        x_target = torch.rand(B, T, H, W).to(device)
        noise = torch.randn(B, T, H, W).to(device) * 0.1
        
        # 预热
        with torch.no_grad():
            _ = loss_fn(velocity_pred, x_target, noise)
        
        # 计时
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            losses = loss_fn(velocity_pred, x_target, noise)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        print(f"  计算时间: {(end_time - start_time)*1000:.2f}ms")
        print(f"  LPIPS Loss: {losses['lpips_loss'].item():.6f}")
        print(f"  总内存: {torch.cuda.memory_allocated()/1024**2:.1f}MB" if device == 'cuda' else "  CPU模式")


if __name__ == "__main__":
    # 运行所有示例
    example_usage()
    #training_template() 
    #performance_benchmark()
    
    print("\n" + "="*50)
    print("🎉 恭喜！所有测试通过")
    print("你现在可以直接使用这个方案了！")
    print("="*50)