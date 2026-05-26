  
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.ours.RWKV_model import VRWKV_Bottleneck
from  models.ours.encoder import OFLMedSeg
from  models.ours.WaveletGuidedSkipConnection import WaveletGuidedSkipConnection
class TemporalAttention(nn.Module):
    """A Temporal Attention block with Cross-Attention"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()                 # GELU
        self.cross_attention = CrossAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, query, key, value):
        if self.attn_shortcut:
            shortcut = query.clone()
        x = self.proj_1(query)
        x = self.activation(x)
        x = self.cross_attention(x, key, value)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x

class CrossAttentionModule(nn.Module):
    """Cross-Attention Module for Temporal Attention"""

    def __init__(self, dim, kernel_size, reduction=16):
        super().__init__()
        self.query_proj = nn.Conv2d(dim, dim, 1)  # 1x1 conv for query projection
        self.key_proj = nn.Conv2d(dim, dim, 1)    # 1x1 conv for key projection
        self.value_proj = nn.Conv2d(dim, dim, 1)  # 1x1 conv for value projection

        self.attn_conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)  # Depth-wise conv
        self.output_proj = nn.Conv2d(dim, dim, 1)  # 1x1 conv for output projection

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction,bias=False),  # Reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim,bias=False),  # Expansion
            nn.Sigmoid()
        )

    def forward(self, query, key, value):
        """
        Args:
            query: Tensor of shape (B, C, H, W) - the query input
            key: Tensor of shape (B, C, H, W) - the key input
            value: Tensor of shape (B, C, H, W) - the value input
        """
        # Project query, key, and value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Compute attention weights
        attn_weights = torch.softmax(torch.einsum('bchw,bcij->bhwij', query, key), dim=-1)  # Attention map
        attn_output = torch.einsum('bhwij,bcij->bchw', attn_weights, value)  # Weighted sum of values

        # Apply depth-wise convolution for spatial refinement
        attn_output = self.attn_conv(attn_output)

        # SE (Squeeze-and-Excitation) operation
        b, c, _, _ = attn_output.size()
        se_atten = self.avg_pool(attn_output).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)

        # Combine SE attention with the output
        return se_atten * attn_output





class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


# def shift(dim):
#             x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
#             x_cat = torch.cat(x_shift, 1)
#             x_cat = torch.narrow(x_cat, 2, self.pad, H)
#             x_cat = torch.narrow(x_cat, 3, self.pad, W)
#             return x_cat


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
def swish(x):
    
    return x * torch.sigmoid(x)


# class TimeEmbedding(nn.Module):
#     def __init__(self, T, d_model, dim):
#         assert d_model % 2 == 0
#         super().__init__()
#         emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
#         emb = torch.exp(-emb)
#         pos = torch.arange(T).float()
#         emb = pos[:, None] * emb[None, :]
#         assert list(emb.shape) == [T, d_model // 2]
#         emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
#         assert list(emb.shape) == [T, d_model // 2, 2]
#         emb = emb.view(T, d_model)

#         self.timembedding = nn.Sequential(
#             nn.Embedding.from_pretrained(emb),
#             nn.Linear(d_model, dim),
#             Swish(),
#             nn.Linear(dim, dim),
#         )
#         self.initialize()

#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)

#     def forward(self, t):
#         emb = self.timembedding(t)
#         return emb

class TimeEmbedding(nn.Module):
    def __init__(self, d_model, dim):
        super().__init__()
        self.d_model = d_model
        
        # 不再需要T参数，因为可以处理任意时间值
        
        # 频率参数，控制正弦波的周期
        self.register_buffer('freq', 
                             torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model))
        
        # 后续的MLP层
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim),
            nn.SiLU(),  # 使用SiLU替代Swish
            nn.Linear(dim, dim),
        )
        
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        # t是[0,1)范围内的连续浮点数
        # 扩展维度以匹配频率参数
        t = t.unsqueeze(-1)  # [batch_size, 1]
        
        # 计算正弦和余弦位置编码
        args = t * self.freq  # [batch_size, d_model//2]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [batch_size, d_model]
        
        # 通过MLP处理嵌入
        return self.mlp(embedding)
    
class ImprovedContinuousTimeEmbedding(nn.Module):
    """改进的连续时间编码 - 针对流模型优化"""
    def __init__(self, d_model, dim):
        super().__init__()
        self.d_model = d_model
        
        # 使用更密集的频率分布，对[0,1]范围优化
        freqs = torch.linspace(0.1, 100, d_model // 2)
        self.register_buffer('freqs', freqs)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # 添加LayerNorm提高稳定性
        )
        
        # 初始化
        self._initialize()
        
    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, t):
        # 确保t在[0,1]范围内
        t = torch.clamp(t, 0, 1)
        t = t.unsqueeze(-1)  # [batch_size, 1]
        
        # 使用更适合[0,1]范围的编码
        args = t * self.freqs * 2 * math.pi  # [batch_size, d_model//2]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return self.mlp(embedding)
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x
    
class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=Swish
        grid_eps=0.02
        grid_range=[-1, 1]

        self.fc1 = KANLinear(
                    in_features,
                    hidden_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()

        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4.,drop_path=0.,norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, dim),
        )

        self.kan = kan(in_features=dim, hidden_features=mlp_hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, temb):

        temb = self.temb_proj(temb)
        x = self.drop_path(self.kan(self.norm2(x), H, W))
        x = x + temb.unsqueeze(1)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.GroupNorm(32, dim)
        # self.relu = Swish()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = swish(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Conv2d(h_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]


class D_SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32,in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(32,in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
             nn.GroupNorm(32,h_ch),
            Swish()
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )
    def forward(self, input,temb):
        return self.conv(input) + self.temb_proj(temb)[:,:,None, None]

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, h_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, h_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(h_ch, h_ch, 3, stride=1, padding=1),
        )
        if in_ch != h_ch:
            self.shortcut = nn.Conv2d(in_ch, h_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(h_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h



class ConditionGuidedSpatialTransformFusion(nn.Module):
    def __init__(self, channels, reduction=16, max_offset=0.2):
        super().__init__()
        # 条件特征生成空间位移场
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, 3, padding=1),  # 输出Δx, Δy
            nn.Tanh()  # 保证位移在[-1, 1]之间
        )
        self.max_offset = max_offset  # 控制最大位移比例

        # 融合层（可自定义）
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # 1. 生成位移场
        offset = self.offset_conv(x2)  # [B, 2, H, W]
        offset = offset * self.max_offset  # 控制最大偏移

        # 2. 构造采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x1.device),
            torch.linspace(-1, 1, W, device=x1.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        # offset: [B, 2, H, W] -> [B, H, W, 2]
        offset = offset.permute(0, 2, 3, 1)
        sampling_grid = grid + offset  # [B, H, W, 2]

        # 3. 对主特征做空间采样
        x1_warped = F.grid_sample(x1, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        # 4. 融合
        out = self.fuse(torch.cat([x1_warped, x2], dim=1))
        return out

import torch.fft as fft
def Fourier_filter(x, threshold, scale):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered


             

class FreeFusion(nn.Module):
    def __init__(self, channels, s, b):
        super().__init__()
        self.s=s
        self.b=b
        self.c=channels
        # 融合层（可自定义）
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x1, x2):
        x1[:,:self.c//2] = x1[:,:self.c//2] * self.b
        x2 = Fourier_filter(x2, threshold=1, scale=self.s)
        
         # 4. 融合
        out = self.fuse(torch.cat([x1, x2], dim=1))
        return out


class ConditionFusion(nn.Module):
    def __init__(self, channels=[64, 128, 128, 128, 256, 320]):
        super().__init__()
        # 使用 ModuleList 存储针对每个尺度的融合块
        self.fusion_blocks = nn.ModuleList()

        for ch in channels:
            # 针对每个尺度构建一个融合层
            # 输入: ch * 2 (因为是拼接)
            # 输出: ch (降维回原始通道数，保持网络宽度一致)
            block = nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True) # 如果你的主干网用的是 SiLU/Swish，这里也可以换
            )
            self.fusion_blocks.append(block)

    def forward(self, features_1, features_2):
        """
        Args:
            features_1: 列表，包含6个 Tensor，通道对应 [64, 128, ...]
            features_2: 列表，包含6个 Tensor，结构同上
        Returns:
            fused_features: 列表，包含融合后的6个 Tensor
        """
        fused_features = []
        
        # 使用 zip 同时遍历：条件1特征、条件2特征、对应的融合层
        for f1, f2, block in zip(features_1, features_2, self.fusion_blocks):
            # 1. 在通道维度拼接 [B, C, H, W] + [B, C, H, W] -> [B, 2C, H, W]
            cat_x = torch.cat([f1, f2], dim=1)
            
            # 2. 卷积融合并降维 -> [B, C, H, W]
            out = block(cat_x)
            
            fused_features.append(out)
            
        return fused_features

class UKan_Hybrid(nn.Module):
    def __init__(self, T,in_ch,out_ch, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index h of bound'
        tdim = ch * 4
        self.time_embedding = TimestepEmbedder(tdim)
        attn = []
        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record hput channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            h_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            h_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, out_ch, 3, stride=1, padding=1)
        )

        # 
        embed_dims = [128,256, 320]
        norm_layer = nn.LayerNorm
        dpr = [0.0, 0.0, 0.0]
        self.patch_embed3 = OverlapPatchEmbed(img_size=64 // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=64 // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])

        self.kan_block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1],  mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer),
            VRWKV_Bottleneck(
        n_embd=embed_dims[1],
        n_layer=8,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )])

        self.kan_block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2],  mlp_ratio=1, drop_path=dpr[1], norm_layer=norm_layer),
            VRWKV_Bottleneck(
        n_embd=embed_dims[2],
        n_layer=12,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )])

        self.kan_dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer),
            VRWKV_Bottleneck(
        n_embd=embed_dims[1],
        n_layer=8,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )])

        self.decoder1 = D_SingleConv(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_SingleConv(embed_dims[1], embed_dims[0])  


        self.encoder=OFLMedSeg()
        # self.encoder2=OFLMedSeg(input_channels=in_ch)
        # self.condition_fusion = ConditionFusion([64, 128, 128, 128, 256, 320])
      

        self.CGSTF0=ConditionGuidedSpatialTransformFusion(channels=64)
        self.CGSTF1=ConditionGuidedSpatialTransformFusion(channels=128)
        self.CGSTF2=ConditionGuidedSpatialTransformFusion(channels=128)
      


        #walete
        self.wgc3=WaveletGuidedSkipConnection(in_channels=128)
        self.wgc4=WaveletGuidedSkipConnection(in_channels=256)


        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t,condition,**kwargs):
        # Timestep embedding
        temb = self.time_embedding(t)

        #condition 
        enc=self.encoder(condition)
       
        # Downsampling
        h = self.head(x)
        hs = [h]

        i=0
        for layer in self.downblocks:
            h = layer(h, temb)
            if i in [1]:
                m = self.CGSTF0(h,enc[0])
                hs.append(m)
               
            elif i in [4]:
                m = self.CGSTF1(h,enc[1])
                hs.append(m)
            
            elif i in [7]:
                m = self.CGSTF2(h,enc[2])
                hs.append(m)

            else:
                hs.append(h)
            i=i+1
           

        #t3 = self.CGSTF3(h,enc[-3])
        t3=h
        

        B = x.shape[0]
        h, H, W = self.patch_embed3(h)
 
        for i, blk in enumerate(self.kan_block1):
            h = blk(h, H, W, temb)
            
        h = self.norm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #t4 = self.CGSTF4(h,enc[-2])
        t4=h
        

        h, H, W= self.patch_embed4(h)
       
        for i, blk in enumerate(self.kan_block2):
            h = blk(h, H, W, temb)
        h = self.norm4(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()    #[8, 320, 9, 9]
        h=h+enc[-1]



        ### Stage 4
        h = swish(F.interpolate(self.decoder1(h, temb), scale_factor=(2,2), mode ='bilinear'))

        h = self.wgc4(t4,h,enc[-2])

        _, _, H, W = h.shape
        h = h.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.kan_dblock1):
            h = blk(h, H, W, temb)
            

            
        ### Stage 3
        h = self.dnorm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   #[8, 256, 18, 18]



        h = swish(F.interpolate(self.decoder2(h, temb),scale_factor=(2,2),mode ='bilinear'))   #[8, 128, 36, 36]
        h = self.wgc3(t3,h,enc[-3])




        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)          #[8, 128, 72, 72] 3   

           
        h = self.tail(h)

        assert len(hs) == 0
        return h




from thop import profile
cuda_idx = 0
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)
if __name__ == '__main__':
    model = UKan_Hybrid(
        T=1000,in_ch=20,out_ch=20, ch=64, ch_mult=[1, 2, 2, 2], attn=[],
        num_res_blocks=2, dropout=0.1).cuda()
    x=torch.rand(1,20,128,128).cuda()
    y=torch.rand(1,20,128,128).cuda()
    t = torch.rand(x.size(0)).cuda()
    flops, params = profile(model, inputs=(y,t,x))
    # out=model(y,t,x)
    # print(out.shape)
    params_in_million = params / 1e6  # \u53c2\u6570\u91cf\u6362\u7b97\u4e3a\u767e\u4e07
    flops_in_billion = flops / 1e9  # FLOPs \u6362\u7b97\u4e3a\u5341\u4ebf
    print(f"Total Parameters: {params_in_million:.3f} M")
    print(f"Total FLOPs: {flops_in_billion:.3f} G")




