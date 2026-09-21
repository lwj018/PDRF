"""Condition encoder (paper Fig. 3).

A 6-level feature pyramid over the historical context C, whose multi-scale
outputs are refined by the Feature Communication Module (FCM):
information propagates bidirectionally across pyramid scales so that
local features are modulated by global context.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class FeatureAlignment(nn.Module):
    """特征对齐模块 - 处理不同尺度特征的对齐"""
    def __init__(self, in_channels, out_channels):
        super(FeatureAlignment, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, target_size):
        x = self.conv(x)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class CrossScaleAttention(nn.Module):
    """跨尺度注意力机制 - 修复版"""
    def __init__(self, channels, num_levels):
        super(CrossScaleAttention, self).__init__()
        self.channels = channels
        self.num_levels = num_levels
        
        # 简化的注意力机制
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels * num_levels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_levels, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * (num_levels - 1), channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, features_aligned):
        """
        features_aligned: List of aligned features with same spatial size [B, C, H, W]
        """
        if len(features_aligned) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} features, got {len(features_aligned)}")
        
        # 确保所有特征具有相同的空间尺寸
        target_size = features_aligned[0].shape[2:]
        aligned_features = []
        for feat in features_aligned:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 拼接所有特征 [B, C*L, H, W]
        concat_features = torch.cat(aligned_features, dim=1)
        
        # 计算注意力权重 [B, L, H, W]
        attention_weights = self.attention_conv(concat_features)
        
        # 应用注意力权重
        enhanced_features = []
        for i, (feat, weight) in enumerate(zip(aligned_features, attention_weights.unbind(1))):
            # weight: [B, H, W] -> [B, 1, H, W]
            weight = weight.unsqueeze(1)
            
            # 融合其他层的信息
            other_features = [f for j, f in enumerate(aligned_features) if j != i]
            if other_features:
                other_concat = torch.cat(other_features, dim=1)
                other_fused = self.fusion_conv(other_concat)
                enhanced = feat + self.gamma * weight * other_fused
            else:
                enhanced = feat
                
            enhanced_features.append(enhanced)
        
        return enhanced_features


class MultiDirectionalFusion(nn.Module):
    """多方向特征融合 - Top-down, Bottom-up, Lateral"""
    def __init__(self, channels_list):
        super(MultiDirectionalFusion, self).__init__()
        self.num_levels = len(channels_list)
        self.channels_list = channels_list
        
        # Top-down pathway
        self.top_down_convs = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.top_down_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[i+1], channels_list[i], 1),
                    nn.BatchNorm2d(channels_list[i]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Bottom-up pathway
        self.bottom_up_convs = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.bottom_up_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[i], channels_list[i+1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels_list[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for channels in channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Fusion weights
        self.fusion_weights = nn.ModuleList()
        for channels in channels_list:
            self.fusion_weights.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, 3, 1),  # 3 for top-down, bottom-up, lateral
                    nn.Softmax(dim=1)
                )
            )
    
    def forward(self, features):
        # Initialize pathways
        td_features = [None] * self.num_levels  # top-down
        bu_features = [None] * self.num_levels  # bottom-up
        lateral_features = []
        
        # Lateral pathway
        for i, feature in enumerate(features):
            lateral_features.append(self.lateral_convs[i](feature))
        
        # Top-down pathway
        td_features[-1] = lateral_features[-1]  # 最顶层直接使用lateral特征
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                self.top_down_convs[i](td_features[i+1]), 
                size=lateral_features[i].shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            td_features[i] = upsampled
        
        # Bottom-up pathway
        bu_features[0] = lateral_features[0]  # 最底层直接使用lateral特征
        for i in range(1, self.num_levels):
            downsampled = self.bottom_up_convs[i-1](bu_features[i-1])
            # 调整到目标尺寸
            if downsampled.shape[2:] != lateral_features[i].shape[2:]:
                downsampled = F.interpolate(
                    downsampled,
                    size=lateral_features[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            bu_features[i] = downsampled
        
        # Adaptive fusion
        fused_features = []
        for i in range(self.num_levels):
            # 确保所有特征都存在且尺寸匹配
            if td_features[i] is None:
                td_features[i] = lateral_features[i]
            if bu_features[i] is None:
                bu_features[i] = lateral_features[i]
                
            # 准备三路特征
            paths = [td_features[i], bu_features[i], lateral_features[i]]
            
            # 计算融合权重 [B, 3, 1, 1]
            weights = self.fusion_weights[i](lateral_features[i])  
            
            # 加权融合 - 修复广播问题
            fused = torch.zeros_like(lateral_features[i])
            for j, (w, path) in enumerate(zip(weights.unbind(1), paths)):
                # w的形状: [B, 1, 1] -> [B, 1, 1, 1]用于广播
                w = w.unsqueeze(1)  # [B, 1, 1, 1]
                fused = fused + w * path
            
            fused_features.append(fused)
        
        return fused_features


class FeatureCommunicationModule(nn.Module):
    """特征交流整合模块"""
    def __init__(self, channels_list, enable_cross_attention=True, enable_multi_fusion=True):
        super(FeatureCommunicationModule, self).__init__()
        self.channels_list = channels_list
        self.num_levels = len(channels_list)
        self.enable_cross_attention = enable_cross_attention
        self.enable_multi_fusion = enable_multi_fusion
        
        # 特征对齐模块 - 将所有特征对齐到中间尺度
        self.alignment_modules = nn.ModuleList()
        middle_channels = channels_list[len(channels_list)//2]
        for channels in channels_list:
            self.alignment_modules.append(
                FeatureAlignment(channels, middle_channels)
            )
        
        # 跨尺度注意力
        if self.enable_cross_attention:
            self.cross_attention = CrossScaleAttention(middle_channels, self.num_levels)
        
        # 多方向融合
        if self.enable_multi_fusion:
            self.multi_fusion = MultiDirectionalFusion(channels_list)
        
        # 特征恢复模块 - 将对齐后的特征恢复到原始通道数
        self.recovery_modules = nn.ModuleList()
        for channels in channels_list:
            self.recovery_modules.append(
                nn.Sequential(
                    nn.Conv2d(middle_channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 最终增强模块
        self.enhancement_modules = nn.ModuleList()
        for channels in channels_list:
            self.enhancement_modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.Sigmoid()  # 注意力权重
                )
            )
    
    def forward(self, getpre):
        """
        输入: getpre - 每一层的输出特征列表
        输出: 增强后的特征列表
        """
        original_features = getpre
        enhanced_features = []
        
        # Step 1: 多方向融合 (在原始尺度空间)
        if self.enable_multi_fusion:
            multi_fused = self.multi_fusion(original_features)
        else:
            multi_fused = original_features
        
        # Step 2: 特征对齐到中间尺度
        if self.enable_cross_attention:
            middle_size = original_features[len(original_features)//2].shape[2:]
            aligned_features = []
            for i, feature in enumerate(multi_fused):
                aligned = self.alignment_modules[i](feature, middle_size)
                aligned_features.append(aligned)
            
            # Step 3: 跨尺度注意力交流
            attended_features = self.cross_attention(aligned_features)
            
            # Step 4: 恢复到原始通道数和尺度
            for i, (attended, original) in enumerate(zip(attended_features, original_features)):
                # 恢复通道数
                recovered = self.recovery_modules[i](attended)
                
                # 恢复空间尺度
                if recovered.shape[2:] != original.shape[2:]:
                    recovered = F.interpolate(
                        recovered, 
                        size=original.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Step 5: 特征增强
                attention_weight = self.enhancement_modules[i](recovered)
                enhanced = original + recovered * attention_weight
                enhanced_features.append(enhanced)
        else:
            # 如果不使用跨尺度注意力，直接增强
            for i, feature in enumerate(multi_fused):
                attention_weight = self.enhancement_modules[i](feature)
                enhanced = original_features[i] + feature * attention_weight
                enhanced_features.append(enhanced)
        
        return enhanced_features


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class CoordinateAttention(nn.Module):
    """坐标注意力机制 - 比SE更强的空间感知能力"""
    def __init__(self, channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 分别在H和W方向进行池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # 拼接并降维
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分离并生成注意力权重
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return identity * a_h * a_w


class FCSB(nn.Module):
    def __init__(self, dim, outdim, mlp_ratio=2, drop_path=0.2, use_se=True, is_deep=False):
        super(FCSB, self).__init__()
        if is_deep:

            kernel_size = 7
            padding = 3
        else:
            kernel_size = 3
            padding = 1
        # 定义可学习的参数
        self.learnable_param = nn.Parameter(torch.randn(1))
        self.dwconv = ConvBN(dim, dim, kernel_size=kernel_size, stride=1, padding=padding, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, outdim, kernel_size=1, with_bn=True)
        self.dwconv2 = ConvBN(outdim, outdim, kernel_size=kernel_size, stride=1, padding=padding, groups=outdim,
                              with_bn=False)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.out = ConvBN(dim, outdim, kernel_size=1, with_bn=False)
        self.use_se = use_se
        if self.use_se:
            self.se = CoordinateAttention(outdim)

    def forward(self, x):
        input = self.out(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        # x = self.act(x1) - x2
        x = self.act(x1) - (x2 * self.learnable_param)
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        if self.use_se:
            x = self.se(x)
        return x


class ConditionEncoder(nn.Module):
    def __init__(self, input_channels=5, c_list=[64,128,128,128,256,320]):  # [16,32,64,96,128,128]
        super().__init__()

        # Encoders
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0],3,1,1),
        )
        self.encoder2 = nn.Sequential(
            FCSB(c_list[0], c_list[1]),

        )
        self.encoder3 = nn.Sequential(
            FCSB(c_list[1], c_list[2]),

        )
        self.encoder4 = nn.Sequential(
            FCSB(c_list[2], c_list[3]),

        )
        self.encoder5 = nn.Sequential(
            FCSB(c_list[3], c_list[4]),

        )
        self.encoder6 = nn.Sequential(
            FCSB(c_list[4], c_list[5]),

        )
        
        self.ebn1 = nn.GroupNorm(32, c_list[0])
        self.ebn2 = nn.GroupNorm(32, c_list[1])
        self.ebn3 = nn.GroupNorm(32, c_list[2])
        self.ebn4 = nn.GroupNorm(32, c_list[3])
        self.ebn5 = nn.GroupNorm(32, c_list[4])
        self.ebn6 = nn.GroupNorm(32, c_list[5])


        self.feature_communication = FeatureCommunicationModule(
            channels_list=c_list,
            enable_cross_attention=True,
            enable_multi_fusion=True
        )
      
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        getpre = []
        out =self.ebn1(self.encoder1(x))
        getpre.append(out)
        out = F.gelu(F.max_pool2d(out, 2, 2))

        out = self.ebn2(self.encoder2(out))
        getpre.append(out)
        out = F.gelu(F.max_pool2d(out, 2, 2))

        out = self.ebn3(self.encoder3(out))
        getpre.append(out)
        out = F.gelu(F.max_pool2d(out, 2, 2))

        out = self.ebn4(self.encoder4(out))
        getpre.append(out)
        out = F.gelu(F.max_pool2d(out, 2, 2))

        out = self.ebn5(self.encoder5(out))
        getpre.append(out)
        out = F.gelu(F.max_pool2d(out, 2, 2))

        out = self.ebn6(self.encoder6(out))
        getpre.append(out)

        enhanced_getpre = self.feature_communication(getpre)


        return enhanced_getpre

FCM = FeatureCommunicationModule
