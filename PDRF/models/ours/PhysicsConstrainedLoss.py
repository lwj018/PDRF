import torch


class PhysicsConstrainedLoss:
    """雷达预测的物理约束损失"""
    
    def __init__(self, 
                 mass_weight=0.1,      # 质量守恒权重
                 gradient_weight=0.05,  # 梯度平滑权重
                 spectrum_weight=0.1,   # 频谱一致性权重
                 flow_weight=0.15):     # 光流一致性权重
        self.mass_weight = mass_weight
        self.gradient_weight = gradient_weight
        self.spectrum_weight = spectrum_weight
        self.flow_weight = flow_weight
    
    def mass_conservation_loss(self, pred, target):
        """质量守恒：预测的总反射率应接近真实值"""
        # 按batch和时间维度计算总和
        pred_sum = pred.sum(dim=(-2, -1))  # [B, T]
        target_sum = target.sum(dim=(-2, -1))
        return torch.nn.functional.l1_loss(pred_sum, target_sum)
    
    def gradient_smoothness_loss(self, pred):
        """空间梯度平滑约束，避免不自然跳变"""
        # Sobel算子计算梯度
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        # 梯度的方差不应过大
        grad_var = dx.var() + dy.var()
        return grad_var
    
    def spectrum_consistency_loss(self, pred, target):
        """频谱一致性：保持正确的空间频率分布"""
        # FFT到频域
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        # 对比幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return torch.nn.functional.l1_loss(pred_mag, target_mag)
    
    def optical_flow_consistency_loss(self, pred, target):
        """光流一致性：预测序列的运动应该合理"""
        if pred.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # 计算预测帧间的光流特征（简化版：帧差）
        pred_diff = pred[:, 1:] - pred[:, :-1]  # [B, T-1, H, W]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # 运动模式应该相似
        return torch.nn.functional.l1_loss(pred_diff, target_diff)
    
    def __call__(self, pred, target):
        """综合物理约束损失"""
        loss = 0.0
        
        if self.mass_weight > 0:
            loss += self.mass_weight * self.mass_conservation_loss(pred, target)
        
        if self.gradient_weight > 0:
            loss += self.gradient_weight * self.gradient_smoothness_loss(pred)
        
        if self.spectrum_weight > 0:
            loss += self.spectrum_weight * self.spectrum_consistency_loss(pred, target)
        
        if self.flow_weight > 0:
            loss += self.flow_weight * self.optical_flow_consistency_loss(pred, target)
        
        return loss