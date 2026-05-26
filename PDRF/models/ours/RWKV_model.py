import math
import os
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.utils.cpp_extension import load
T_MAX = 640

wkv_cuda = load(name="wkv", sources=[os.path.join(os.getcwd(),'models','ours','wkv_op.cpp'), os.path.join(os.getcwd(),'models','ours', 'wkv_cuda.cu')],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])


# def q_shift(input, shift_pixel=1, gamma=0.25, patch_resolution=None, add_residual=False):
#     """
#     Q-Shift 操作的 4D 版本，输入形状为 (B, C, H, W)。
    
#     Args:
#         input: 输入张量，形状为 (B, C, H, W)。
#         shift_pixel: 每个方向的移动距离。
#         gamma: 每个方向的通道数比例。
#         add_residual: 是否添加残差连接，默认为 True。
    
#     Returns:
#         经过 Q-Shift 操作后的张量，如果 add_residual=True，则带有残差连接。
#     """
#     assert gamma <= 1/4
#     B, N, C = input.shape
#     if patch_resolution is None:
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             patch_resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

#     input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    
#     B, C, H, W = input.shape
#     output = torch.zeros_like(input)

#     # 计算每个方向的通道数
#     gamma_C = int(C * gamma)

#     # 右移
#     output[:, 0:gamma_C, :, shift_pixel:W] = input[:, 0:gamma_C, :, 0:W-shift_pixel]
#     # 左移
#     output[:, gamma_C:2*gamma_C, :, 0:W-shift_pixel] = input[:, gamma_C:2*gamma_C, :, shift_pixel:W]
#     # 下移
#     output[:, 2*gamma_C:3*gamma_C, shift_pixel:H, :] = input[:, 2*gamma_C:3*gamma_C, 0:H-shift_pixel, :]
#     # 上移
#     output[:, 3*gamma_C:4*gamma_C, 0:H-shift_pixel, :] = input[:, 3*gamma_C:4*gamma_C, shift_pixel:H, :]
#     # 保留剩余通道
#     output[:, 4*gamma_C:, ...] = input[:, 4*gamma_C:, ...]

#     # 如果 add_residual=True，则添加残差连接
#     if add_residual:
#         output = input + output

#     return output.flatten(2).transpose(1,2)


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    if patch_resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            patch_resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    input = input
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)
def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())
class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy', k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = {
            shift_mode: q_shift}
        if shift_pixel > 0:
            self.shift_func = self.shift_mode[shift_mode]
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.value.scale_init = 1

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = {
            shift_mode: q_shift}
        if shift_pixel > 0:
            self.shift_func = self.shift_mode[shift_mode]
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

        self.key.scale_init = 1

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class VRWKV_Bottleneck(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 drop_path=0., k_norm=True):
        """
        Bottleneck module for (B, N, C) input shape.
        
        Args:
            n_embd (int): Embedding dimension (C).
            n_layer (int): Total number of layers in the model.
            layer_id (int): Current layer ID.
            shift_mode (str): Shift mode for spatial mixing.
            channel_gamma (float): Gamma for channel mixing.
            shift_pixel (int): Number of pixels to shift.
            hidden_rate (int): Hidden dimension multiplier for FFN.
            init_mode (str): Initialization mode ('fancy', 'local', 'global').
            drop_path (float): Drop path rate.
            k_norm (bool): Whether to use key normalization.
        """
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.flatten = nn.Flatten(2)
        # Spatial Mixing (Attention-like mechanism)
        self.spatial_mix = VRWKV_SpatialMix(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            init_mode=init_mode,
            k_norm=k_norm
        )

        # Channel Mixing (FFN-like mechanism)
        self.channel_mix = VRWKV_ChannelMix(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            hidden_rate=hidden_rate,
            init_mode=init_mode,
            k_norm=k_norm
        )

        # LayerNorm for post-processing
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, n_embd),
        )

    def forward(self, x,H,W,temb, patch_resolution=None):
        """
        Forward pass for the bottleneck.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            patch_resolution (tuple): Resolution of the patches (H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        # Spatial Mixing (Attention-like)
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1,2)
        x = x + self.drop_path(self.spatial_mix(self.ln1(x), patch_resolution))

        # Channel Mixing (FFN-like)
        x = x + self.drop_path(self.channel_mix(self.ln2(x), patch_resolution))
        
        # if len(x.shape) == 3:
            
        #     out = x.transpose(1, 2).reshape(B, C, H, W)

        temb = self.temb_proj(temb)
        return x+ temb.unsqueeze(1)
    

# from thop import profile
# cuda_idx = 0
# device = torch.device('cuda:' + str(cuda_idx))
# torch.cuda.set_device(device)
# if __name__ == '__main__':
#     model = VRWKV_Bottleneck(
#         n_embd=320,
#         n_layer=12,
#         layer_id=0,
#         shift_mode='q_shift',
#         channel_gamma=1/4,
#         shift_pixel=1,
#         hidden_rate=4,
#         init_mode='fancy',
#         drop_path=0.1,
#         k_norm=True
#     ).cuda()
#     x=torch.rand(4,320,18,18).cuda()
#     out=model(x)
#     print(out.shape)




