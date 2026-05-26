import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class WaveletGuidedSkipConnection(nn.Module):
    """
    小波引导的跳跃连接模块
    
    Args:
        in_channels: 输入特征通道数
        wavelet_type: 小波类型 ('db4', 'haar', 'bior2.2' 等)
        reduction_ratio: 通道注意力的降维比例
    """
    
    def __init__(self, in_channels, wavelet_type='db4', reduction_ratio=8):
        super(WaveletGuidedSkipConnection, self).__init__()
        
        self.in_channels = in_channels
        self.wavelet_type = wavelet_type
        
        # 小波系数处理网络
        self.wavelet_processor = WaveletProcessor(in_channels, wavelet_type)
        
        # 条件引导注意力模块
        self.condition_attention = ConditionGuidedAttention(
            in_channels, reduction_ratio
        )
        
        # 特征融合网络
        self.fusion_net = FeatureFusionNet(in_channels)
        
        # 最终输出调整
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_feat, decoder_feat, condition_feat):
        """
        Args:
            encoder_feat: 主网络编码器特征 (B, C, H, W)
            decoder_feat: 主网络解码器特征 (B, C, H, W)  
            condition_feat: 条件编码器特征 (B, C, H, W)
        
        Returns:
            fused_feat: 融合后的特征 (B, C, H, W)
        """
        B, C, H, W = encoder_feat.shape
        
        # 1. 对条件特征进行小波分解，生成引导信息
        wavelet_guidance = self.wavelet_processor(condition_feat)
        
        # 2. 基于小波引导生成注意力权重
        attention_weights = self.condition_attention(
            encoder_feat, decoder_feat, wavelet_guidance
        )
        
        # 3. 特征融合
        fused_feat = self.fusion_net(
            encoder_feat, decoder_feat, attention_weights
        )
        
        # 4. 最终输出处理
        output = self.output_conv(fused_feat)
        
        return output


class WaveletProcessor(nn.Module):
    """小波变换处理器"""
    
    def __init__(self, channels, wavelet_type='db4'):
        super(WaveletProcessor, self).__init__()
        
        self.wavelet_type = wavelet_type
        self.channels = channels
        
        # 低频分量处理
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, 1, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 高频分量处理 (LH, HL, HH)
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels//2, 3, 1, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 小波特征融合
        self.wavelet_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # 生成0-1的引导权重
        )
    
    def dwt_2d(self, x):
        """2D小波变换"""
        B, C, H, W = x.shape
        
        # 转换为numpy进行小波变换
        x_np = x.detach().cpu().numpy()
        
        # 存储小波系数
        coeffs_list = []
        
        for b in range(B):
            batch_coeffs = []
            for c in range(C):
                # 对每个通道进行2D小波变换
                coeffs = pywt.dwt2(x_np[b, c], self.wavelet_type, mode='periodization')
                batch_coeffs.append(coeffs)
            coeffs_list.append(batch_coeffs)
        
        # 重新组织小波系数
        LL_list, (LH_list, HL_list, HH_list) = [], ([], [], [])
        
        for b in range(B):
            LL_batch, LH_batch, HL_batch, HH_batch = [], [], [], []
            for c in range(C):
                LL, (LH, HL, HH) = coeffs_list[b][c]
                LL_batch.append(LL)
                LH_batch.append(LH)
                HL_batch.append(HL)
                HH_batch.append(HH)
            
            LL_list.append(np.stack(LL_batch, axis=0))
            LH_list.append(np.stack(LH_batch, axis=0))
            HL_list.append(np.stack(HL_batch, axis=0))
            HH_list.append(np.stack(HH_batch, axis=0))
        
        # 转换回tensor
        device = x.device
        LL = torch.tensor(np.stack(LL_list, axis=0), dtype=x.dtype, device=device)
        LH = torch.tensor(np.stack(LH_list, axis=0), dtype=x.dtype, device=device)
        HL = torch.tensor(np.stack(HL_list, axis=0), dtype=x.dtype, device=device)
        HH = torch.tensor(np.stack(HH_list, axis=0), dtype=x.dtype, device=device)
        
        return LL, LH, HL, HH
    
    def forward(self, condition_feat):
        """
        对条件特征进行小波分解并生成引导信息
        """
        # 小波分解
        LL, LH, HL, HH = self.dwt_2d(condition_feat)
        
        # 处理低频分量 (主要结构信息)
        low_freq_feat = self.low_freq_conv(LL)
        
        # 处理高频分量 (细节信息)
        high_freq_combined = torch.cat([LH, HL, HH], dim=1)
        high_freq_feat = self.high_freq_conv(high_freq_combined)
        
        # 上采样到原始尺寸
        _, _, H, W = condition_feat.shape
        low_freq_feat = F.interpolate(low_freq_feat, size=(H, W), mode='bilinear', align_corners=False)
        high_freq_feat = F.interpolate(high_freq_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 融合低频和高频特征
        wavelet_feat = torch.cat([low_freq_feat, high_freq_feat], dim=1)
        guidance_weights = self.wavelet_fusion(wavelet_feat)
        
        return guidance_weights


class ConditionGuidedAttention(nn.Module):
    """条件引导的注意力机制"""
    
    def __init__(self, channels, reduction_ratio=8):
        super(ConditionGuidedAttention, self).__init__()
        
        reduced_channels = max(channels // reduction_ratio, 1)
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels * 3, reduced_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, reduced_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, 1, 0),
            nn.Sigmoid()
        )
        
        # 小波引导的权重生成
        self.wavelet_weight_gen = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, encoder_feat, decoder_feat, wavelet_guidance):
        """
        生成注意力权重
        """
        # 特征concatenation
        combined_feat = torch.cat([encoder_feat, decoder_feat, wavelet_guidance], dim=1)
        
        # 空间注意力
        spatial_attn = self.spatial_attention(combined_feat)
        
        # 通道注意力
        channel_attn = self.channel_attention(combined_feat)
        
        # 小波引导权重
        wavelet_weights = self.wavelet_weight_gen(wavelet_guidance)
        
        return {
            'spatial': spatial_attn,
            'channel': channel_attn,
            'wavelet': wavelet_weights
        }


class FeatureFusionNet(nn.Module):
    """特征融合网络"""
    
    def __init__(self, channels):
        super(FeatureFusionNet, self).__init__()
        
        # 编码器特征处理
        self.encoder_proc = nn.Conv2d(channels, channels, 3, 1, 1)
        
        # 解码器特征处理  
        self.decoder_proc = nn.Conv2d(channels, channels, 3, 1, 1)
        
        # 自适应融合权重
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_feat, decoder_feat, attention_weights):
        """
        融合编码器和解码器特征
        """
        # 应用注意力权重
        enc_weighted = encoder_feat * attention_weights['spatial'] * attention_weights['channel']
        dec_weighted = decoder_feat * attention_weights['wavelet']
        
        # 特征处理
        enc_processed = self.encoder_proc(enc_weighted)
        dec_processed = self.decoder_proc(dec_weighted)
        
        # 计算自适应融合权重
        fusion_input = torch.cat([enc_processed, dec_processed], dim=1)
        adaptive_weights = self.fusion_weights(fusion_input)
        
        # 加权融合
        fused = enc_processed * adaptive_weights + dec_processed * (1 - adaptive_weights)
        
        # 最终特征增强
        final_input = torch.cat([fused, encoder_feat], dim=1)
        output = self.final_fusion(final_input)
        
        return output


# 使用示例
if __name__ == "__main__":
    # 创建模块
    skip_connection = WaveletGuidedSkipConnection(in_channels=64)
    
    # 模拟输入
    B, C, H, W = 2, 64, 128, 128
    encoder_feat = torch.randn(B, C, H, W)
    decoder_feat = torch.randn(B, C, H, W) 
    condition_feat = torch.randn(B, C, H, W)
    
    # 前向传播
    output = skip_connection(encoder_feat, decoder_feat, condition_feat)
    
    print(f"输入形状: {encoder_feat.shape}")
    print(f"输出形状: {output.shape}")
    print("模块参数量:", sum(p.numel() for p in skip_connection.parameters()))