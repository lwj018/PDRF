import random
import torch.nn as nn
import torch

from models.ours.schedulers.rf.Lpiploss import create_rectified_flow_loss
from models.ours.ps_loss2 import RainfallPSLoss

from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat
from .time_sampler import TimeSampler

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def soft_csi_loss(
    x_pred: torch.Tensor,
    x_target: torch.Tensor,
    threshold: float = 0.1,
    k: float = 10.0,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
):
    """
    x_pred, x_target: [B, C, T, H, W] 或兼容形状
    threshold: 降水阈值（按你数据的单位来设）
    k: sigmoid 的斜率，越大越接近硬阈值
    mask: 可选，[B, T, H, W] 或 [B, 1, T, H, W]，1 表示参与 CSI 计算
    """
    # 对 GT 做硬阈值，得到 0/1 的“是否降水”真值
    y_true = (x_target >= threshold).float()

    # 模型输出经过 sigmoid 变成“降水概率”
    # 这里假设 x_pred 跟 x_target 在同一数值尺度
    p_rain = torch.sigmoid(k * (x_pred - threshold))

    # 可选 mask：只在关注区域/时间上计算 CSI
    if mask is not None:
        # 扩到和 x_pred 同维度
        while mask.dim() < x_pred.dim():
            mask = mask.unsqueeze(1)  # [B,T,H,W] -> [B,1,T,H,W]
        p_rain = p_rain * mask
        y_true = y_true * mask

    # 按 batch 维度统计：算每个样本一个 CSI 再取平均
    # x_pred 形状大多是 [B, C, T, H, W]
    dims = tuple(range(1, x_pred.dim()))   # 除 B 外所有维度

    TP = (p_rain * y_true).sum(dim=dims)
    FP = (p_rain * (1.0 - y_true)).sum(dim=dims)
    FN = ((1.0 - p_rain) * y_true).sum(dim=dims)

    csi = TP / (TP + FP + FN + eps)        # [B]
    loss = 1.0 - csi.mean()                # 标量

    return loss


# def gradient_consistency_loss(delta_pred, delta_gt):
#     """
#     仅用残差：delta_pred=velocity_pred, delta_gt=x_target - noise
#     对齐空间梯度（x、y 方向），鼓励残差的结构与真值一致
#     """
#     def grad_xy(x):
#         gx = x[..., :, 1:] - x[..., :, :-1]   # 水平梯度
#         gy = x[..., 1:, :] - x[..., :-1, :]   # 垂直梯度
#         return gx, gy

#     pgx, pgy = grad_xy(delta_pred)
#     tgx, tgy = grad_xy(delta_gt)
#     return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()

def gradient_consistency_loss_improved(
    delta_pred: torch.Tensor,
    delta_gt: torch.Tensor,
    rain_mask: torch.Tensor = None,
    eps: float = 1e-3,
):
    """
    结构导向的梯度一致性：
    - delta_pred: 模型预测的 quantity（比如 velocity_pred）
    - delta_gt:   对应的 ground truth（比如 x_target 或 x_target - noise）
    - rain_mask:  可选，[..., H, W]，1 表示关注区域（如有雨 / 大残差）
    """

    def grad_xy(x):
        # x: [..., H, W]
        gx = x[..., :, 1:] - x[..., :, :-1]   # [..., H,   W-1]
        gy = x[..., 1:, :] - x[..., :-1, :]   # [..., H-1, W  ]
        return gx, gy

    # 1) 计算 pred / gt 的梯度
    pgx, pgy = grad_xy(delta_pred)
    tgx, tgy = grad_xy(delta_gt)

    # 2) 把 gx / gy 裁成共同的内部区域 [H-1, W-1]
    #   pgx: [..., H,   W-1] -> [..., H-1, W-1] （去掉最后一行）
    #   pgy: [..., H-1, W  ] -> [..., H-1, W-1] （去掉最后一列）
    pgx_c = pgx[..., :-1, :]     # [..., H-1, W-1]
    pgy_c = pgy[..., :, :-1]     # [..., H-1, W-1]
    tgx_c = tgx[..., :-1, :]
    tgy_c = tgy[..., :, :-1]

    # 3) 误差 + GT 梯度幅值（在同一 shape 上）
    dx_err = torch.sqrt((pgx_c - tgx_c) ** 2 + eps**2)
    dy_err = torch.sqrt((pgy_c - tgy_c) ** 2 + eps**2)
    grad_err = dx_err + dy_err                     # [..., H-1, W-1]

    mag_gt = torch.sqrt(tgx_c**2 + tgy_c**2 + eps**2)  # [..., H-1, W-1]

    # 4) 可选 mask：也要对齐到 [..., H-1, W-1]
    if rain_mask is not None:
        # rain_mask: [..., H, W] -> 取内部区域对齐梯度
        mask_c = rain_mask[..., 1:, 1:]            # [..., H-1, W-1]
        grad_err = grad_err * mask_c
        mag_gt   = mag_gt * mask_c

    # 5) 用 GT 梯度幅值做权重：结构明显的地方惩罚更大
    weighted_err = grad_err * (1.0 + mag_gt)

    return weighted_err.mean()

def get_temporal_weights(T, strategy='linear', device='cuda'):
    """
    生成时序权重 [T]
    T: 帧数（如20）
    strategy: 'linear', 'quadratic', 'exponential', 'sqrt'
    """
    if strategy == 'linear':
        # 线性增长: [0.5, 0.55, 0.6, ..., 1.5]
        weights = torch.linspace(0.5, 2.0, T, device=device)
    elif strategy == 'quadratic':
        # 二次增长: 后期加速增大
        t = torch.linspace(0, 1, T, device=device)
        weights = 0.5 + 1.5 * t**2  # [0.5 -> 2.0]
    elif strategy == 'exponential':
        # 指数增长: 后期权重急剧增大
        t = torch.linspace(0, 1, T, device=device)
        weights = torch.exp(2 * t)  # [1.0 -> 7.39]
        weights = weights / weights[0] * 0.5  # 归一化到 [0.5, ...]
    elif strategy == 'sqrt':
        # 平方根增长: 前期快速增长，后期缓慢
        t = torch.linspace(0, 1, T, device=device)
        weights = 0.5 + 1.5 * torch.sqrt(t)
    return weights

def get_temporal_weights_rain(T, device='cuda',
                              k_head=4,   # 前 k 帧: baseline 很准，权重低一点
                              k_tail=2,   # 最后 k 帧: 极长时距，不要太重
                              w_min=0.5,  # 前几帧权重
                              w_max=2.0   # 中期最高权重
                              ):
    """
    为降雨预测设计的时间权重:
    - 前 k_head 帧: 权重较小 (w_min)
    - 中间帧: 权重从 w_min 平滑上升到 w_max (二次曲线)
    - 最后 k_tail 帧: 略低于峰值，避免过拟合最远期噪声
    
    返回: [T]，平均值归一到 1.0
    """
    assert k_head + k_tail <= T

    # --- 前段：固定低权重 ---
    head = torch.full((k_head,), w_min, device=device)

    # --- 中段：二次上升到 w_max ---
    mid_len = T - k_head - k_tail
    if mid_len > 0:
        t_mid = torch.linspace(0, 1, mid_len, device=device)
        mid = w_min + (w_max - w_min) * (t_mid ** 2)
    else:
        mid = torch.tensor([], device=device)

    # --- 尾段：略微下降，避免极端放大 ---
    if k_tail > 0:
        # 尾段从 w_max 线性减到 (w_min + w_max)/2
        w_tail_end = (w_min + w_max) / 2
        t_tail = torch.linspace(0, 1, k_tail, device=device)
        tail = w_max + (w_tail_end - w_max) * t_tail
    else:
        tail = torch.tensor([], device=device)

    weights = torch.cat([head, mid, tail], dim=0)  # [T]

    # --- 平均归一到 1，不影响整体 loss 尺度 ---
    weights = weights / weights.mean()
    return weights
class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        # time sampler
        sample_method="uniform",
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        transform_scale=1.0,
        scale_temporal=True,
        uniform_over_threshold=None,
        drop_condition=None,
        x_cond_weight=1,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.time_sampler = TimeSampler(
            sample_method=sample_method,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            transform_scale=transform_scale,
            scale_temporal=scale_temporal,
            uniform_over_threshold=uniform_over_threshold,
        )
        self.drop_condition = drop_condition
        self.x_cond_weight = x_cond_weight  # the weight to use for i2v and v2v condition

        self.ps_loss=RainfallPSLoss(
        temporal_patch_size=3,      # T=9时，3个时间步为一个patch最合适
        spatial_patch_size=24,      # 空间patch大小，适合288分辨率
        temporal_stride=2,          # 时间步长，约67%重叠
        spatial_stride=12,          # 空间步长，50%重叠
        adaptive_patching=True,     # 开启自适应patching
        lambda_temporal=1.0,        # 时间损失权重
        lambda_spatial=1.0          # 空间损失权重
    )   
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lpips_loss_fn = create_rectified_flow_loss(lpips_weight=0.1, use_gpu=(device=='cuda'))


    def training_losses(
        self,
        model,
        x_start,
        condition,
        x_sl,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
        x_gt=None,
        mask_index=None,
        noise_disable_threshold=None,
        text_uncond_prob=None,
        x_noisy_ref=None,
    ):
        """
        Compute training losses for a single timestep.
        Note: t is int tensor in [0, num_timesteps-1] (step index)，
        连续时间 t_cont = 1 - t / num_timesteps ∈ (0, 1].
        """

        y_null = None

        # ---------- 条件 mask（你的原逻辑基本不动） ----------
        if mask_index is not None and len(mask_index) > 0:  # i2v and v2v
            num_frames = x_start.shape[2]
            x_cond_mask = torch.zeros_like(x_start, device=x_start.device)
            x_cond_mask[:, :, mask_index, :, :] = 1.0
            x_noisy_ref = x_noisy_ref if x_noisy_ref is not None else x_start
            x_cond = x_noisy_ref * x_cond_mask
        else:
            x_cond_mask = torch.zeros_like(x_start, device=x_start.device)
            x_cond = x_start * x_cond_mask

        if t is None:
            t = self.time_sampler.sample(x_start, self.num_timesteps, model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        # 可选：大 t 段不用噪声（原逻辑）
        if noise_disable_threshold is not None:
            no_noise_mask = t > noise_disable_threshold
            x_start[no_noise_mask] = x_gt[no_noise_mask]

        # ---------- 前向加噪，得到 x_t ----------
        x_t = self.add_noise(x_start, noise, t)

        # 可选：只对 mask 区域加噪
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}

        # ======================================================
        # 关键 1：模型现在直接预测“干净图像 x_pred”，而不是速度
        # ======================================================
        x_pred = model(
            x_t,
            t,          # 仍然把 step index 传进去，模型里你爱怎么 embed 都行
            condition,
            # cond=x_cond,
            # cond_mask=x_cond_mask,
            # mask_index=mask_index,
            # y_null=y_null,
            # text_uncond_prob=text_uncond_prob,
            # **model_kwargs,
        )
        # x_pred 的形状应该和 x_start 一样：[B, C, T, H, W]

        # ---------- 掩码权重（保持你的逻辑，只是基于 x_pred 的 shape） ----------
        res_weights = None
        if mask_index is not None and len(mask_index) > 0:
            mask_nums = len(mask_index)
            num_frames = x_start.shape[2]
            new_weight = (num_frames - mask_nums * self.x_cond_weight) / (
                num_frames - mask_nums
            )  # scale to ensure sum is same
            res_weights = (
                torch.ones_like(x_pred).to(x_pred.device).to(x_pred.dtype) * new_weight
            )  # 非 mask 帧的权重
            res_weights[:, :, mask_index, :, :] = 1 * self.x_cond_weight  # 被 mask 帧的权重

        # x 的 GT：有 x_gt 用 x_gt，没有就用 x_start（你原来的逻辑）
        x_target = x_gt if x_gt is not None else x_start

        # ---------- 统一构造连续时间 t_cont ∈ (0,1]，和 add_noise 完全一致 ----------
        # add_noise: timepoints = 1 - timesteps / num_timesteps
        # 这里的 t 就是离散 step index ∈ [0, num_timesteps-1]
        t_cont = 1.0 - t.float() / float(self.num_timesteps)  # [B]

        # broadcast 到和 x_t 同维度
        if t_cont.dim() == 1:
            t_broadcast = t_cont.view(-1, *([1] * (x_t.ndim - 1)))
        else:
            t_broadcast = t_cont
            while t_broadcast.ndim < x_t.ndim:
                t_broadcast = t_broadcast.view(*t_broadcast.shape, 1)

        # (1 - t_cont)，避免除 0
        one_minus_t = (1.0 - t_broadcast).clamp(min=1e-3)

        # ======================================================
        # 关键 2：从 x_pred / x_target + x_t 里，计算“速度” v_pred / v_target
        # 路径：x_t = t_cont * x + (1 - t_cont) * noise
        # 推导：v = (x - x_t) / (1 - t_cont) = x - noise（和你原来的目标一致）
        # ======================================================
        v_pred   = (x_pred   - x_t)      / one_minus_t
        v_target = (x_target - x_t)      / one_minus_t  
        
        # 2) 新增：半拉格朗日预测帧的“物理速度”
        # x_sl: 半拉格朗日预测出来的帧（和 x_target 同时间的预测）
        v_sl = (x_sl - x_t) / one_minus_t

        # 可选：时间权重（你现在没真正用到）
        temporal_weights = get_temporal_weights(
            x_pred.shape[1], strategy='sqrt', device=x_pred.device
        )
        temporal_weights = temporal_weights.view(1, x_pred.shape[1], 1, 1)

        # ======================================================
        # 关键 3：loss 在 v_pred / v_target 上算（flow matching）
        # ======================================================
        if weights is None:
            if res_weights is not None:
                loss_mse = mean_flat(
                    res_weights * (v_pred - v_target).pow(2), mask=mask
                )
            else:
                loss_mse = mean_flat(
                    (v_pred - v_target).pow(2), mask=mask
                ).sum()   # 如果想稳定一点可以改成 .mean()

            # 如果后面你要恢复梯度一致性正则，这里用 v_pred / v_target 即可
            # loss_gc = gradient_consistency_loss_improved(
            #     v_pred,
            #     v_target,
            #     (x_target.abs() > 0.1).float(),
            # )
            # loss = loss_mse + 0.1 * loss_gc
            loss = loss_mse

        else:
            weight = _extract_into_tensor(weights, t, x_target.shape)
            if res_weights is not None:
                loss = mean_flat(
                    res_weights * weight * (v_pred - v_target).pow(2),
                    mask=mask,
                )
            else:
                loss = mean_flat(
                    weight * (v_pred - v_target).pow(2),
                    mask=mask,
                )
        
        # 4) 物理蒸馏 loss（软约束）
        lambda_sl = 0.1  # 或 0.05 / 0.01，根据实验调
        loss_sl = mean_flat((v_pred - v_sl).pow(2), mask=mask).sum()
        terms["loss"] = loss+ lambda_sl * loss_sl
        return terms
    
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()

        timesteps: step index ∈ [0, num_timesteps-1]
        连续时间 t_cont = 1 - timesteps / num_timesteps ∈ (0,1]
        路径: x_t = t_cont * original_samples + (1 - t_cont) * noise
        """
        # 离散步 -> [0,1] 再翻转
        timepoints = timesteps.float() / self.num_timesteps      # [0, 1)
        timepoints = 1 - timepoints                              # (0, 1]

        # broadcast 到 [B, C, T, H]，最后一个维度 W 靠广播
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3])

        return timepoints * original_samples + (1 - timepoints) * noise





