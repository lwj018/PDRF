import torch
import torch.nn as nn
import torch.nn.functional as F

class RainfallPSLoss(nn.Module):
    """
    Patch-wise Structural Loss for Rainfall Prediction
    适用于时空数据 (B, T, H, W)
    """
    def __init__(self, temporal_patch_size=4, spatial_patch_size=8, 
                 temporal_stride=2, spatial_stride=4, 
                 adaptive_patching=True, lambda_temporal=1.0, lambda_spatial=1.0):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_stride = temporal_stride
        self.spatial_stride = spatial_stride
        self.adaptive_patching = adaptive_patching
        self.lambda_temporal = lambda_temporal
        self.lambda_spatial = lambda_spatial
        self.kl_loss = nn.KLDivLoss(reduction='none')
        
    def fourier_based_temporal_patching(self, x):
        """
        基于傅里叶变换的自适应时间patching
        x: [B, T, H, W]
        """
        B, T, H, W = x.shape
        
        # 对时间维度进行FFT分析，取空间平均
        x_temporal = x.mean(dim=(2, 3))  # [B, T]
        
        # 添加数值稳定性检查
        if torch.isnan(x_temporal).any() or torch.isinf(x_temporal).any():
            temporal_patch_len = min(self.temporal_patch_size, T//2)
            temporal_stride = max(temporal_patch_len // 2, 1)
            return temporal_patch_len, temporal_stride
            
        x_fft = torch.fft.rfft(x_temporal, dim=1)  # [B, T//2+1]
        frequency_magnitude = torch.abs(x_fft).mean(0)  # [T//2+1]
        frequency_magnitude[0] = 0.0  # 去除直流分量
        
        # 找到主频率
        if frequency_magnitude.sum() > 0:
            top_index = torch.argmax(frequency_magnitude)
            period = max(T // max(top_index.item(), 1), 2)
            temporal_patch_len = min(period // 2, self.temporal_patch_size, T//2)
            temporal_stride = max(temporal_patch_len // 2, 1)
        else:
            temporal_patch_len = min(self.temporal_patch_size, T//2)
            temporal_stride = max(temporal_patch_len // 2, 1)
            
        return temporal_patch_len, temporal_stride
    
    def create_temporal_patches(self, x, patch_len, stride):
        """
        创建时间维度的patches
        x: [B, T, H, W] -> patches: [B, num_patches, patch_len, H, W]
        """
        B, T, H, W = x.shape
        num_patches = (T - patch_len) // stride + 1
        
        patches = []
        for i in range(num_patches):
            start_idx = i * stride
            end_idx = start_idx + patch_len
            patch = x[:, start_idx:end_idx, :, :]  # [B, patch_len, H, W]
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # [B, num_patches, patch_len, H, W]
        return patches
    
    def create_spatial_patches(self, x, patch_size, stride):
        """
        创建空间维度的patches
        x: [B, T, H, W] -> patches: [B, T, num_patches_h, num_patches_w, patch_size, patch_size]
        """
        B, T, H, W = x.shape
        
        # 确保patch参数有效
        patch_size = min(patch_size, H, W)
        stride = min(stride, patch_size)
        
        num_patches_h = (H - patch_size) // stride + 1
        num_patches_w = (W - patch_size) // stride + 1
        
        patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        # [B, T, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = patches.contiguous().view(B, T, num_patches_h, num_patches_w, patch_size, patch_size)
        
        return patches
    
    def calculate_patch_statistics(self, patches):
        """
        计算patch的统计信息
        """
        # 计算均值
        patch_mean = torch.mean(patches, dim=-1, keepdim=True)
        if patches.dim() > patch_mean.dim():
            # 对于spatial patches，需要在最后两个维度上计算均值
            patch_mean = torch.mean(patches, dim=(-2, -1), keepdim=True)
        
        # 计算方差
        if patches.dim() > patch_mean.dim():
            patch_var = torch.var(patches.view(*patches.shape[:-2], -1), dim=-1, keepdim=True, unbiased=False)
        else:
            patch_var = torch.var(patches, dim=-1, keepdim=True, unbiased=False)
        
        # 添加数值稳定性
        patch_std = torch.sqrt(torch.clamp(patch_var, min=1e-8))
        
        return patch_mean, patch_var, patch_std
    
    def temporal_structural_loss(self, true_patches, pred_patches):
        """
        时间维度的结构损失
        patches: [B, num_patches, patch_len, H, W]
        """
        # 展平空间维度进行计算
        B, num_patches, patch_len, H, W = true_patches.shape
        true_flat = true_patches.view(B, num_patches, patch_len, -1)  # [B, num_patches, patch_len, H*W]
        pred_flat = pred_patches.view(B, num_patches, patch_len, -1)
        
        # 对每个patch计算统计量
        true_mean = torch.mean(true_flat, dim=2, keepdim=True)  # [B, num_patches, 1, H*W]
        pred_mean = torch.mean(pred_flat, dim=2, keepdim=True)
        
        true_var = torch.var(true_flat, dim=2, keepdim=True, unbiased=False)
        pred_var = torch.var(pred_flat, dim=2, keepdim=True, unbiased=False)
        # 添加数值稳定性
        true_std = torch.sqrt(torch.clamp(true_var, min=1e-8))
        pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-8))
        
        # 计算协方差
        true_centered = true_flat - true_mean
        pred_centered = pred_flat - pred_mean
        covariance = torch.mean(true_centered * pred_centered, dim=2, keepdim=True)
        
        # 1. 线性相关性损失 - 添加数值稳定性
        correlation = (covariance + 1e-8) / (true_std * pred_std + 1e-8)
        # 裁剪相关系数到合理范围
        correlation = torch.clamp(correlation, -1.0 + 1e-6, 1.0 - 1e-6)
        corr_loss = (1.0 - correlation).mean()
        
        # 2. 分布差异损失 (KL散度) - 添加数值稳定性
        # 限制输入范围避免softmax溢出
        true_flat_clipped = torch.clamp(true_flat, -10, 10)
        pred_flat_clipped = torch.clamp(pred_flat, -10, 10)
        
        true_softmax = torch.softmax(true_flat_clipped, dim=2)
        pred_log_softmax = torch.log_softmax(pred_flat_clipped, dim=2)
        
        # 确保概率分布的数值稳定性
        true_softmax = torch.clamp(true_softmax, min=1e-8)
        
        var_loss = self.kl_loss(pred_log_softmax, true_softmax).sum(dim=2).mean()
        # 限制KL散度的上界
        var_loss = torch.clamp(var_loss, max=100.0)
        
        # 3. 均值损失
        mean_loss = torch.abs(true_mean - pred_mean).mean()
        
        return corr_loss, var_loss, mean_loss
    
    def spatial_structural_loss(self, true_patches, pred_patches):
        """
        空间维度的结构损失
        patches: [B, T, num_patches_h, num_patches_w, patch_size, patch_size]
        """
        # 展平patch内部
        B, T, nh, nw, ps1, ps2 = true_patches.shape
        true_flat = true_patches.view(B, T, nh, nw, -1)  # [B, T, nh, nw, ps1*ps2]
        pred_flat = pred_patches.view(B, T, nh, nw, -1)
        
        # 对每个spatial patch计算统计量
        true_mean = torch.mean(true_flat, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred_flat, dim=-1, keepdim=True)
        
        true_var = torch.var(true_flat, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred_flat, dim=-1, keepdim=True, unbiased=False)
        # 添加数值稳定性
        true_std = torch.sqrt(torch.clamp(true_var, min=1e-8))
        pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-8))
        
        # 计算协方差
        true_centered = true_flat - true_mean
        pred_centered = pred_flat - pred_mean
        covariance = torch.mean(true_centered * pred_centered, dim=-1, keepdim=True)
        
        # 1. 线性相关性损失 - 添加数值稳定性
        correlation = (covariance + 1e-8) / (true_std * pred_std + 1e-8)
        # 裁剪相关系数到合理范围
        correlation = torch.clamp(correlation, -1.0 + 1e-6, 1.0 - 1e-6)
        corr_loss = (1.0 - correlation).mean()
        
        # 2. 分布差异损失 - 添加数值稳定性
        # 限制输入范围避免softmax溢出
        true_flat_clipped = torch.clamp(true_flat, -10, 10)
        pred_flat_clipped = torch.clamp(pred_flat, -10, 10)
        
        true_softmax = torch.softmax(true_flat_clipped, dim=-1)
        pred_log_softmax = torch.log_softmax(pred_flat_clipped, dim=-1)
        
        # 确保概率分布的数值稳定性
        true_softmax = torch.clamp(true_softmax, min=1e-8)
        
        var_loss = self.kl_loss(pred_log_softmax, true_softmax).sum(dim=-1).mean()
        # 限制KL散度的上界
        var_loss = torch.clamp(var_loss, max=100.0)
        
        # 3. 均值损失
        mean_loss = torch.abs(true_mean - pred_mean).mean()
        
        return corr_loss, var_loss, mean_loss
    
    def gradient_based_weighting(self, losses, model_params):
        """
        基于梯度的动态权重调整
        """
        corr_loss, var_loss, mean_loss = losses
        
        if model_params is None:
            # 如果没有提供模型参数，使用固定权重
            return torch.tensor(1.0, device=corr_loss.device), torch.tensor(1.0, device=corr_loss.device), torch.tensor(1.0, device=corr_loss.device)
        
        try:
            # 计算各损失的梯度 - 简化梯度计算避免二阶梯度问题
            corr_grad = torch.autograd.grad(corr_loss, model_params, 
                                          retain_graph=True, create_graph=False)[0]
            var_grad = torch.autograd.grad(var_loss, model_params, 
                                         retain_graph=True, create_graph=False)[0]
            mean_grad = torch.autograd.grad(mean_loss, model_params, 
                                          retain_graph=True, create_graph=False)[0]
            
            # 计算梯度范数 - 添加数值稳定性
            corr_norm = torch.clamp(corr_grad.norm().detach(), min=1e-8)
            var_norm = torch.clamp(var_grad.norm().detach(), min=1e-8)
            mean_norm = torch.clamp(mean_grad.norm().detach(), min=1e-8)
            
            # 平均梯度范数
            avg_norm = (corr_norm + var_norm + mean_norm) / 3.0
            
            # 动态权重 - 限制权重范围避免极端值
            alpha = torch.clamp(avg_norm / corr_norm, 0.1, 10.0)
            beta = torch.clamp(avg_norm / var_norm, 0.1, 10.0)
            gamma = torch.clamp(avg_norm / mean_norm, 0.1, 10.0)
            
            return alpha, beta, gamma
            
        except Exception:
            # 如果梯度计算失败，返回固定权重
            return torch.tensor(1.0, device=corr_loss.device), torch.tensor(1.0, device=corr_loss.device), torch.tensor(1.0, device=corr_loss.device)
    
    def forward(self, true, pred, model_params=None):
        """
        前向传播
        true, pred: [B, T, H, W]
        model_params: 模型参数用于梯度计算（可选）
        """
        B, T, H, W = true.shape
        
        # 添加输入检查和裁剪
        if torch.isnan(true).any() or torch.isnan(pred).any() or torch.isinf(true).any() or torch.isinf(pred).any():
            # 如果输入包含异常值，返回零损失避免训练中断
            device = true.device
            return {
                'total_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'temporal_loss': torch.tensor(0.0, device=device),
                'spatial_loss': torch.tensor(0.0, device=device),
                'temporal_components': (torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)),
                'spatial_components': (torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)),
                'temporal_weights': (torch.tensor(1.0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)),
                'spatial_weights': (torch.tensor(1.0, device=device), torch.tensor(1.0, device=device), torch.tensor(1.0, device=device))
            }
        
        # 1. 自适应时间patching
        if self.adaptive_patching:
            temporal_patch_len, temporal_stride = self.fourier_based_temporal_patching(true)
        else:
            temporal_patch_len = self.temporal_patch_size
            temporal_stride = self.temporal_stride
        
        # 2. 创建时间patches
        true_temporal_patches = self.create_temporal_patches(true, temporal_patch_len, temporal_stride)
        pred_temporal_patches = self.create_temporal_patches(pred, temporal_patch_len, temporal_stride)
        
        # 3. 计算时间结构损失
        temporal_losses = self.temporal_structural_loss(true_temporal_patches, pred_temporal_patches)
        
        # 4. 创建空间patches
        true_spatial_patches = self.create_spatial_patches(true, self.spatial_patch_size, self.spatial_stride)
        pred_spatial_patches = self.create_spatial_patches(pred, self.spatial_patch_size, self.spatial_stride)
        
        # 5. 计算空间结构损失
        spatial_losses = self.spatial_structural_loss(true_spatial_patches, pred_spatial_patches)
        
        # 6. 时间维度的动态权重
        temporal_weights = self.gradient_based_weighting(temporal_losses, model_params)
        temporal_ps_loss = (temporal_weights[0] * temporal_losses[0] + 
                           temporal_weights[1] * temporal_losses[1] + 
                           temporal_weights[2] * temporal_losses[2])
        
        # 7. 空间维度的动态权重
        spatial_weights = self.gradient_based_weighting(spatial_losses, model_params)
        spatial_ps_loss = (spatial_weights[0] * spatial_losses[0] + 
                          spatial_weights[1] * spatial_losses[1] + 
                          spatial_weights[2] * spatial_losses[2])
        
        # 8. 最终损失
        total_loss = (self.lambda_temporal * temporal_ps_loss + 
                     self.lambda_spatial * spatial_ps_loss)
        
        # 添加最终数值检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=true.device, requires_grad=True)
            
        # return {
        #     'total_loss': total_loss,
        #     'temporal_loss': temporal_ps_loss,
        #     'spatial_loss': spatial_ps_loss,
        #     'temporal_components': temporal_losses,
        #     'spatial_components': spatial_losses,
        #     'temporal_weights': temporal_weights,
        #     'spatial_weights': spatial_weights
        # }

        return total_loss

# 针对T=9, 288x288降雨数据的专用配置
def example_usage_T9_288():
    """针对T=9, 288x288降雨数据的使用示例"""
    
    # 针对T=9的最优参数配置
    rainfall_loss = RainfallPSLoss(
        temporal_patch_size=3,      # T=9时，3个时间步为一个patch最合适
        spatial_patch_size=24,      # 空间patch大小，适合288分辨率
        temporal_stride=2,          # 时间步长，约67%重叠
        spatial_stride=12,          # 空间步长，50%重叠
        adaptive_patching=True,     # 开启自适应patching
        lambda_temporal=1.0,        # 时间损失权重
        lambda_spatial=1.0          # 空间损失权重
    )
    
    # T=9的降雨数据 [B, T, H, W]
    batch_size, time_steps, height, width = 2, 9, 288, 288
    true_rainfall = torch.randn(batch_size, time_steps, height, width)
    pred_rainfall = torch.randn(batch_size, time_steps, height, width)
    
    print(f"数据维度: {true_rainfall.shape}")
    print(f"T=9的patches分析:")
    print(f"  - 时间patches: {(time_steps - 3) // 2 + 1} = 4个")
    print(f"  - 空间patches: {((288 - 24) // 12 + 1) ** 2} = 529个")
    print(f"  - 总patch pairs: 4 × 529 = 2,116对")
    
    # 计算损失
    loss_dict = rainfall_loss(true_rainfall, pred_rainfall, model_params=None)
    
    print(f"\n损失结果:")
    print(f"  - Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  - Temporal Loss: {loss_dict['temporal_loss'].item():.4f}")
    print(f"  - Spatial Loss: {loss_dict['spatial_loss'].item():.4f}")
    
    return loss_dict




# example_usage_T9_288()