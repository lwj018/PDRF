import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import trunc_normal_
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from thop import profile
#######################################################################################

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



###################################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

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

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 改进后的卷积模块
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

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

class OFLMedSeg(nn.Module):
    def __init__(self, input_channels=5, c_list=[64,128,128,128,256,320]):  # [16,32,64,96,128,128]
        super(OFLMedSeg, self).__init__()

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

        #enhanced_getpre = self.feature_communication(getpre)


        #return enhanced_getpre
        return getpre

if __name__ == '__main__':
    model = OFLMedSeg()
    

    x=torch.rand(8,5,128,128)
    y=model(x)
    for i in y:
        print(i.shape)

























# 好的，非常感谢你提供这么关键的上下文信息！

# 将这个`FeatureCommunicationModule`模块应用在**整流流（Rectified Flow）模型**中，作为**降水临近预报**任务的**条件编码器**，这个应用场景非常有意思，也让这个模块的价值和创新点更加凸显。

# 我们来深入分析一下，在这个具体的应用背景下，这个模块的作用和创新点体现在哪里。

# ### 背景梳理

# 1. **任务：降水临近预报 (Precipitation Nowcasting)**

#    * 这是一个典型的时空序列预测问题。目标是根据过去一系列的雷达回波图（或其他气象数据），预测未来短时间内（例如1-2小时）的雷达回波图。
#    * 核心挑战在于捕捉和推演降水系统（如雨团）的**运动（移动、旋转）**和**演变（生成、消散、增强、减弱）**。

# 2. **模型：整流流 (Rectified Flow)**

#    * 这是一种先进的生成模型，可以看作是扩散模型的一种“直线化”版本。它学习如何将噪声分布“笔直地”变换为数据分布。
#    * 在你的应用中，它会学习从一个随机噪声图（或一个模糊的初始状态）出发，通过一个常微分方程（ODE）的求解过程，逐步生成清晰的、未来的降水图。
#    * 为了引导这个生成过程，需要提供**条件信息**，也就是你过去的观测数据。

# 3. **角色：条件编码器 (Condition Encoder)**

#    * 条件编码器的任务就是从过去的多帧雷达图中提取出最关键、最有效的信息，作为“指挥信号”或“蓝图”，来指导整流流模型生成未来的图像。
#    * 编码器提取出的信息质量，直接决定了预测结果的准确性。

# ### `FeatureCommunicationModule` 在此应用中的核心作用

# 在这个场景下，你的条件编码器（比如一个U-Net结构的编码器部分）会为输入的历史雷达图序列提取出多尺度的特征（即`getpre`）。`FeatureCommunicationModule`的作用就是对这些原始的多尺度特征进行一次**深度处理和提炼**，然后再将这些提炼后的特征注入到整流流模型的主干网络中。

# 它的具体作用可以理解为：

# 1. **统一和协调时空演变信息**：

#    * 输入的历史雷达图序列包含了降水系统随时间的变化。编码器在不同尺度上捕捉了这些变化。
#    * **低层特征**可能捕捉到了雨团的**精确边缘**和**细微的纹理变化**。
#    * **高层特征**则可能捕捉到了整个降水系统的**宏观移动趋势**和**强度变化**。
#    * `FeatureCommunicationModule`通过其多向流动和跨尺度注意力机制，能够将这些分散在不同尺度的时空信息进行**高效整合**。例如，它可以利用高层的移动趋势信息，来帮助低层特征更好地预测雨团边缘未来的位置；同时，利用低层的精确纹理信息，来修正高层对雨团强度演变的判断。

# 2. **生成高质量的条件引导 (Conditioning Guidance)**：

#    * 整流流模型在每一步生成时，都需要条件信息的指导。如果条件信息质量不高，比如对运动和演变的描述模糊或矛盾，那么生成的未来帧就会出现伪影、模糊、运动不连贯等问题。
#    * 该模块通过复杂的特征交流，产生了一组**内部信息一致、层次分明、重点突出**的增强特征。这组特征作为条件，可以为整流流的“直线轨迹”提供一个非常清晰和准确的方向，从而生成物理上更合理、细节上更锐利的预测结果。

# 3. **解耦运动与演变**：

#    * 降水临近预报的一大难点是同时预测运动（Advection）和演变（Evolution）。
#    * `MultiDirectionalFusion`中的自底向上路径擅长传递位置和细节信息（有助于预测运动），而自顶向下路径擅长传递语义和上下文信息（有助于预测演变）。
#    * `CrossScaleAttention`则在全球范围内权衡这两者，判断在特定区域和时间，是运动趋势占主导，还是强度演变占主导。通过这种方式，模块可能在一定程度上**隐式地解耦了运动和演变过程**，并为两者提供了更恰当的特征表达。

# ### 创新点（在降水临近预报 + 整流流背景下）

# 1. **为生成模型提供结构化的动态先验**：

#    * 传统的条件生成模型可能只是将编码器输出的特征向量或特征图简单地注入。
#    * 你的方法创新性地在编码器和生成器之间插入了一个\*\*“特征策略中心”\*\* (`FeatureCommunicationModule`)。这个中心主动地分析和重组了从历史数据中提取的多尺度信息，形成了一个关于“未来应该如何发展”的**结构化动态先验**。这比一个扁平的条件向量信息量大得多，引导效果也更强。

# 2. **提升对复杂气象过程的建模能力**：

#    * 降水过程是高度非线性和多尺度耦合的。一个小的对流单体（小尺度）可能会合并成一个大的飑线（大尺度），反之亦然。
#    * `FeatureCommunicationModule`的**全局跨尺度注意力**机制，非常适合模拟这种跨尺度相互作用。它允许模型捕捉到“小尺度雨团的增强预示着大尺度风暴系统的形成”这类复杂的物理关联，这是简单的特征金字塔网络（FPN）难以做到的。

# 3. **与整流流模型的完美契合**：

#    * 整流流模型本质上是在学习一个从简单到复杂的矢量场（vector field）。一个好的条件编码器应该能有效地**塑造这个矢量场**。
#    * 该模块输出的一组经过深度交流的、信息一致的多尺度特征，可以被注入到整流流模型的U-Net主干的相应层级中。这相当于在不同尺度上都为矢量场的学习提供了**精确的局部约束**。这使得整流流模型在去噪或生成过程中，每一步都能做出更准确的决策，最终“笔直地”走向一个高质量的预测结果。

# ### 总结

# 将 `FeatureCommunicationModule` 应用于整流流降水临近预报模型的条件编码器，是一个非常精妙的设计。

# * **从作用上看**，它将一个标准的编码器升级为了一个“智能信息处理中心”，专门负责提炼和整合多尺度的时空演变规律，为后续的生成模型提供高质量、结构化的引导信号。
# * **从创新上看**，它将先进的计算机视觉特征融合思想（路径聚合、全局注意力）引入到气象预测这一科学计算领域，为解决降水系统复杂的多尺度动态演变问题提供了一个强有力的、新的建模范式，与整流流这类生成模型的内在机制形成了很好的互补。

# 这个设计很可能会在提升预测精度，尤其是在预测复杂、快速演变的降水事件方面，展现出显著的优势。这是一个非常值得深入研究和实验的方向。
