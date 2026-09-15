"""Semi-Lagrangian advection prior (paper Sec. 3.2).

A dense motion field u = FlowFB(X_{tau-1}, X_tau) is estimated from the
last two historical frames (Farneback), and k-step backtracing defines
the prior frames  X_SL^(k) = W_k(X_tau; u)  (Eqs. 9-11).
X_SL is mapped into the CRF velocity space as a soft teacher for the
physics-guided velocity regularizer L_phy.
"""

import numpy as np
import torch
from scipy.ndimage import map_coordinates
import cv2


class SemiLagrangianPrior:
    """
    半拉格朗日外推（改良版）
    - 假设 upstream 已将所有帧统一归一化到 [0,1]
    - Farneback 光流使用固定刻度，不做逐帧 min-max
    - 无指数衰减；插值=三次样条；边界=nearest
    - 支持按步数刷新流场，提升长时效稳定性
    """

    def __init__(
        self,
        motion_estimation_method: str = "farneback",   # 'farneback' | 'gradient'
        extrapolation_timestep: int = 5,               # 分钟（占位，若要物理量换算可用）
        border_treatment: str = "nearest",             # 'nearest' | 'reflect' | 'zeros'
        interp_order: int = 3,                         # 3: cubic, 1: bilinear
        smooth_flow_sigma: float = 1.2,                # 光流高斯平滑 sigma；<=0 关闭
        clip_flow_ratio: float = 0.10,                 # 位移限幅= min(H,W)*ratio
        refresh_flow_every: int | None = None          # 每多少步重估一次流场；None/0 关闭
    ):
        self.motion_method = motion_estimation_method
        self.timestep = extrapolation_timestep
        self.border_treatment = border_treatment
        self.interp_order = int(interp_order)
        self.smooth_flow_sigma = float(smooth_flow_sigma)
        self.clip_flow_ratio = float(clip_flow_ratio)
        self.refresh_flow_every = (
            int(refresh_flow_every) if refresh_flow_every and refresh_flow_every > 0 else None
        )

    # ------------------------ 光流估计 ------------------------

    @staticmethod
    def _unit_to_u8(frame01: np.ndarray) -> np.ndarray:
        """已在 [0,1] 的帧 → uint8 固定刻度（避免逐帧 min-max）"""
        f = np.clip(frame01, 0.0, 1.0)
        return (f * 255.0).astype(np.uint8)

    def estimate_motion_farneback(self, frames: np.ndarray) -> np.ndarray:
        """
        用最后两帧估计稠密光流（像素/步）
        frames: [T,H,W]，值域已是 [0,1]
        return: [2,H,W]，(u,v)，右/下为正
        """
        if len(frames) < 2:
            raise ValueError("至少需要2帧来估计运动")

        f1 = self._unit_to_u8(frames[-2].astype(np.float32))
        f2 = self._unit_to_u8(frames[-1].astype(np.float32))

        # Farneback 参数可按需微调
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2,
            None,            # init flow
            0.5,             # pyr_scale
            3,               # levels
            15,              # winsize
            3,               # iterations
            5,               # poly_n
            1.2,             # poly_sigma
            0                # flags
        )  # (H,W,2)

        u, v = flow[..., 0], flow[..., 1]

        # 平滑 + 限幅
        if self.smooth_flow_sigma and self.smooth_flow_sigma > 0:
            u = cv2.GaussianBlur(u, (0, 0), self.smooth_flow_sigma)
            v = cv2.GaussianBlur(v, (0, 0), self.smooth_flow_sigma)

        max_motion = min(frames.shape[1], frames.shape[2]) * self.clip_flow_ratio
        if max_motion > 0:
            u = np.clip(u, -max_motion, max_motion)
            v = np.clip(v, -max_motion, max_motion)

        return np.stack([u, v], axis=0)

    @staticmethod
    def _estimate_motion_gradient(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """简化梯度法（备用）"""
        dt = frame2 - frame1
        dx = ndimage.sobel(frame1, axis=1)
        dy = ndimage.sobel(frame1, axis=0)

        dx_s = ndimage.gaussian_filter(dx, sigma=1.0)
        dy_s = ndimage.gaussian_filter(dy, sigma=1.0)
        dt_s = ndimage.gaussian_filter(dt, sigma=1.0)

        denom = dx_s**2 + dy_s**2 + 1e-6
        u = -dt_s * dx_s / denom
        v = -dt_s * dy_s / denom

        u = ndimage.gaussian_filter(u, sigma=2.0)
        v = ndimage.gaussian_filter(v, sigma=2.0)

        max_motion = min(frame1.shape) * 0.10
        u = np.clip(u, -max_motion, max_motion)
        v = np.clip(v, -max_motion, max_motion)
        return np.array([u, v])

    # ------------------------ 外推主流程 ------------------------

    def extrapolate_frames(self, input_frames: np.ndarray, n_future_frames: int) -> np.ndarray:
        """
        半拉格朗日外推
        input_frames: [T,H,W] 或 [T,C,H,W]，值域已是 [0,1]
        return: [K,H,W] 或 [K,C,H,W]，同值域
        """
        shape = input_frames.shape
        if len(shape) == 4:  # [T,C,H,W]
            T, C, H, W = shape
            outs = []
            for c in range(C):
                outs.append(self._extrapolate_single_channel(input_frames[:, c, :, :], n_future_frames))
            return np.stack(outs, axis=1)  # [K,C,H,W]
        elif len(shape) == 3:  # [T,H,W]
            return self._extrapolate_single_channel(input_frames, n_future_frames)
        else:
            raise ValueError(f"不支持的输入形状: {shape}")

    def _extrapolate_single_channel(self, frames: np.ndarray, K: int) -> np.ndarray:
        """
        单通道外推
        frames: [T,H,W] in [0,1]
        return: [K,H,W]
        """
        T, H, W = frames.shape

        # 初始光流
        if self.motion_method == "farneback":
            uv = self.estimate_motion_farneback(frames)
        else:
            uv = self._estimate_motion_gradient(frames[-2], frames[-1])
        u, v = uv[0], uv[1]

        # 坐标网格
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

        # 边界模式
        if self.border_treatment == "nearest":
            mode = "nearest"; cval = 0.0
        elif self.border_treatment == "reflect":
            mode = "reflect"; cval = 0.0
        elif self.border_treatment == "zeros":
            mode = "constant"; cval = 0.0
        else:
            mode = "nearest"; cval = 0.0

        # 累积生成
        out = []
        last_frame = frames[-1].astype(np.float32)  # 作为 t=0 的源
        prev_frame_for_refresh = frames[-2].astype(np.float32)

        for t in range(1, K + 1):
            # 反向追踪
            x_src = xx - t * u
            y_src = yy - t * v
            x_src = np.clip(x_src, 0, W - 1)
            y_src = np.clip(y_src, 0, H - 1)

            coords = np.array([y_src, x_src])
            # 三次样条 + 合理边界，保护强回波峰值
            ft = map_coordinates(
                last_frame, coords, order=self.interp_order,
                mode=mode, cval=cval, prefilter=False
            )
            out.append(ft)

            # 流场刷新（可选）：每 m 步用最近两帧基线重估 uv
            if self.refresh_flow_every and (t % self.refresh_flow_every == 0) and (t < K):
                # 用“上一基线帧”和“当前基线帧”重估
                prev = last_frame
                curr = ft
                if self.motion_method == "farneback":
                    uv = self.estimate_motion_farneback(np.stack([prev, curr], axis=0))
                else:
                    uv = self._estimate_motion_gradient(prev, curr)
                u, v = uv[0], uv[1]
                prev_frame_for_refresh = prev
                last_frame = curr  # 更新“最近真实/基线帧”
            # 若不刷新，也保持 last_frame 不变（始终追踪自初始 last）

        return np.stack(out, axis=0).astype(np.float32)

    # ------------------------ Torch 包装 ------------------------

    def extrapolate_tensor(self, input_tensor: torch.Tensor, n_future_frames: int) -> torch.Tensor:
        """
        input_tensor: [B,T,H,W] 或 [B,T,C,H,W]，值域 [0,1]
        return:       [B,K,H,W] 或 [B,K,C,H,W]，同值域
        """
        device = input_tensor.device
        shape = input_tensor.shape
        x_np = input_tensor.detach().cpu().numpy()

        outs = []
        for b in range(x_np.shape[0]):
            if len(shape) == 5:   # [B,T,C,H,W]
                fr = x_np[b]      # [T,C,H,W]
            elif len(shape) == 4: # [B,T,H,W]
                fr = x_np[b]      # [T,H,W]
            else:
                raise ValueError(f"不支持的输入形状: {shape}")

            try:
                yb = self.extrapolate_frames(fr, n_future_frames)  # [K,H,W] 或 [K,C,H,W]
            except Exception as e:
                print(f"[SemiLagrangian] Batch {b} 外推失败: {e}")
                # 失败回退：用最后一帧复制
                if len(shape) == 5:
                    last = fr[-1]                                 # [C,H,W]
                    yb = np.tile(last[None], (n_future_frames, 1, 1, 1))
                else:
                    last = fr[-1]                                 # [H,W]
                    yb = np.tile(last[None], (n_future_frames, 1, 1))
            outs.append(yb)

        y_np = np.array(outs)
        y = torch.from_numpy(y_np).float().to(device)
        return y


    @torch.no_grad()
    def get_extrapolated_condition(self, frames_in: torch.Tensor, frames_out_len: int) -> torch.Tensor:
        """
        frames_in:  [B,T_in,H,W] 或 [B,T_in,C,H,W]，值域 [0,1]
        return:     [B,K,H,W]    或 [B,K,C,H,W]，值域 [0,1]
        """
        return self.extrapolate_tensor(frames_in, frames_out_len)

    @torch.no_grad()
    def prepare_training_data(self, frames_in: torch.Tensor, frames_out: torch.Tensor):
        """
        返回 (Y_base, frames_out)
        """
        y_base = self.get_extrapolated_condition(frames_in, frames_out.shape[1])
        return y_base, frames_out

    @staticmethod
    def build_condition(frames_in: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        """
        把 Y_base 并入 condition：cond = concat(ctx, Y_base)（沿时间维）
        - frames_in: [B,T_in,H,W] 或 [B,T_in,C,H,W]
        - y_base:    [B,K,H,W]    或 [B,K,C,H,W]
        return:      [B,T_in+K,H,W] 或 [B,T_in+K,C,H,W]
        """
        assert frames_in.dim() in (4, 5), f"frames_in shape={frames_in.shape}"
        assert y_base.dim() in (4, 5), f"y_base shape={y_base.shape}"
        # 统一通道维
        if frames_in.dim() == 4:
            fi = frames_in
            yb = y_base
            cond = torch.cat([fi, yb], dim=1)  # [B, T_in+K, H, W]
        else:
            fi = frames_in
            yb = y_base
            cond = torch.cat([fi, yb], dim=1)  # [B, T_in+K, C, H, W]
        return cond
