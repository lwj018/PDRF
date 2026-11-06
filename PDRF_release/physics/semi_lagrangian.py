"""Semi-Lagrangian advection prior (paper Sec. 3.2).

A dense motion field u = FlowFB(X_{tau-1}, X_tau) is estimated from the
last two historical frames (Farneback), and k-step backtracing defines
the prior frames  X_SL^(k) = W_k(X_tau; u)  (Eqs. 9-11).
X_SL is mapped into the CRF velocity space as a soft teacher for the
physics-guided velocity regularizer L_phy.
"""

import numpy as np
import torch
import scipy.ndimage as ndimage
from scipy.ndimage import map_coordinates
import cv2


class SemiLagrangianPrior:
    """
    Semi-Lagrangian extrapolation (improved version)
    - Assumes upstream has already normalized all frames to [0, 1]
    - Farneback optical flow uses a fixed scale, no per-frame min-max
    - No exponential decay; interpolation = cubic spline; border = nearest
    - Supports refreshing the flow field every N steps for better
      long-horizon stability
    """

    def __init__(
        self,
        motion_estimation_method: str = "farneback",   # 'farneback' | 'gradient'
        extrapolation_timestep: int = 5,               # minutes (placeholder; use if physical conversion is needed)
        border_treatment: str = "nearest",             # 'nearest' | 'reflect' | 'zeros'
        interp_order: int = 3,                         # 3: cubic, 1: bilinear
        smooth_flow_sigma: float = 1.2,                # Gaussian smoothing sigma for optical flow; <=0 disables
        clip_flow_ratio: float = 0.10,                 # displacement clip = min(H, W) * ratio
        refresh_flow_every: int | None = None          # re-estimate flow every N steps; None/0 disables
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

    # ------------------------ Optical flow estimation ------------------------

    @staticmethod
    def _unit_to_u8(frame01: np.ndarray) -> np.ndarray:
        """Frame already in [0, 1] -> uint8 with a fixed scale (avoids per-frame min-max)"""
        f = np.clip(frame01, 0.0, 1.0)
        return (f * 255.0).astype(np.uint8)

    def estimate_motion_farneback(self, frames: np.ndarray) -> np.ndarray:
        """
        Estimate dense optical flow (pixels/step) from the last two frames.
        frames: [T, H, W], values already in [0, 1]
        return: [2, H, W], (u, v), positive = right/down
        """
        if len(frames) < 2:
            raise ValueError("At least 2 frames are required to estimate motion")

        f1 = self._unit_to_u8(frames[-2].astype(np.float32))
        f2 = self._unit_to_u8(frames[-1].astype(np.float32))

        # Farneback parameters can be tuned as needed
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
        )  # (H, W, 2)

        u, v = flow[..., 0], flow[..., 1]

        # Smooth + clip
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
        """Simplified gradient-based method (fallback)"""
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

    # ------------------------ Extrapolation main pipeline ------------------------

    def extrapolate_frames(self, input_frames: np.ndarray, n_future_frames: int) -> np.ndarray:
        """
        Semi-Lagrangian extrapolation.
        input_frames: [T, H, W] or [T, C, H, W], values already in [0, 1]
        return:       [K, H, W] or [K, C, H, W], same value range
        """
        shape = input_frames.shape
        if len(shape) == 4:  # [T, C, H, W]
            T, C, H, W = shape
            outs = []
            for c in range(C):
                outs.append(self._extrapolate_single_channel(input_frames[:, c, :, :], n_future_frames))
            return np.stack(outs, axis=1)  # [K, C, H, W]
        elif len(shape) == 3:  # [T, H, W]
            return self._extrapolate_single_channel(input_frames, n_future_frames)
        else:
            raise ValueError(f"Unsupported input shape: {shape}")

    def _extrapolate_single_channel(self, frames: np.ndarray, K: int) -> np.ndarray:
        """
        Single-channel extrapolation.
        frames: [T, H, W] in [0, 1]
        return: [K, H, W]
        """
        T, H, W = frames.shape

        # Initial optical flow
        if self.motion_method == "farneback":
            uv = self.estimate_motion_farneback(frames)
        else:
            uv = self._estimate_motion_gradient(frames[-2], frames[-1])
        u, v = uv[0], uv[1]

        # Coordinate grid
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

        # Border mode
        if self.border_treatment == "nearest":
            mode = "nearest"; cval = 0.0
        elif self.border_treatment == "reflect":
            mode = "reflect"; cval = 0.0
        elif self.border_treatment == "zeros":
            mode = "constant"; cval = 0.0
        else:
            mode = "nearest"; cval = 0.0

        # Accumulate generated frames
        out = []
        last_frame = frames[-1].astype(np.float32)  # source at t = 0
        prev_frame_for_refresh = frames[-2].astype(np.float32)

        for t in range(1, K + 1):
            # Backward tracing
            x_src = xx - t * u
            y_src = yy - t * v
            x_src = np.clip(x_src, 0, W - 1)
            y_src = np.clip(y_src, 0, H - 1)

            coords = np.array([y_src, x_src])
            # Cubic spline + sensible border to protect strong echo peaks
            ft = map_coordinates(
                last_frame, coords, order=self.interp_order,
                mode=mode, cval=cval, prefilter=False
            )
            out.append(ft)

            # Flow refresh (optional): re-estimate uv from the two most recent baseline frames every m steps
            if self.refresh_flow_every and (t % self.refresh_flow_every == 0) and (t < K):
                # Re-estimate from the "previous baseline frame" and the "current baseline frame"
                prev = last_frame
                curr = ft
                if self.motion_method == "farneback":
                    uv = self.estimate_motion_farneback(np.stack([prev, curr], axis=0))
                else:
                    uv = self._estimate_motion_gradient(prev, curr)
                u, v = uv[0], uv[1]
                prev_frame_for_refresh = prev
                last_frame = curr  # update "most recent real/baseline frame"
            # If not refreshed, last_frame stays unchanged (always traced back to the initial last frame)

        return np.stack(out, axis=0).astype(np.float32)

    # ------------------------ Torch wrapper ------------------------

    def extrapolate_tensor(self, input_tensor: torch.Tensor, n_future_frames: int) -> torch.Tensor:
        """
        input_tensor: [B, T, H, W] or [B, T, C, H, W], values in [0, 1]
        return:       [B, K, H, W] or [B, K, C, H, W], same value range
        """
        device = input_tensor.device
        shape = input_tensor.shape
        x_np = input_tensor.detach().cpu().numpy()

        outs = []
        for b in range(x_np.shape[0]):
            if len(shape) == 5:   # [B, T, C, H, W]
                fr = x_np[b]      # [T, C, H, W]
            elif len(shape) == 4: # [B, T, H, W]
                fr = x_np[b]      # [T, H, W]
            else:
                raise ValueError(f"Unsupported input shape: {shape}")

            try:
                yb = self.extrapolate_frames(fr, n_future_frames)  # [K, H, W] or [K, C, H, W]
            except Exception as e:
                print(f"[SemiLagrangian] Batch {b} extrapolation failed: {e}")
                # Fallback on failure: repeat the last frame
                if len(shape) == 5:
                    last = fr[-1]                                 # [C, H, W]
                    yb = np.tile(last[None], (n_future_frames, 1, 1, 1))
                else:
                    last = fr[-1]                                 # [H, W]
                    yb = np.tile(last[None], (n_future_frames, 1, 1))
            outs.append(yb)

        y_np = np.array(outs)
        y = torch.from_numpy(y_np).float().to(device)
        return y

    @torch.no_grad()
    def get_extrapolated_condition(self, frames_in: torch.Tensor, frames_out_len: int) -> torch.Tensor:
        """
        frames_in: [B, T_in, H, W] or [B, T_in, C, H, W], values in [0, 1]
        return:    [B, K, H, W]    or [B, K, C, H, W], values in [0, 1]
        """
        return self.extrapolate_tensor(frames_in, frames_out_len)

    @torch.no_grad()
    def prepare_training_data(self, frames_in: torch.Tensor, frames_out: torch.Tensor):
        """
        Returns (Y_base, frames_out)
        """
        y_base = self.get_extrapolated_condition(frames_in, frames_out.shape[1])
        return y_base, frames_out

    @staticmethod
    def build_condition(frames_in: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        """
        Merge Y_base into the condition: cond = concat(ctx, Y_base) (along the time dim)
        - frames_in: [B, T_in, H, W] or [B, T_in, C, H, W]
        - y_base:    [B, K, H, W]    or [B, K, C, H, W]
        return:      [B, T_in+K, H, W] or [B, T_in+K, C, H, W]
        """
        assert frames_in.dim() in (4, 5), f"frames_in shape={frames_in.shape}"
        assert y_base.dim() in (4, 5), f"y_base shape={y_base.shape}"
        # Unify channel dim
        if frames_in.dim() == 4:
            fi = frames_in
            yb = y_base
            cond = torch.cat([fi, yb], dim=1)  # [B, T_in+K, H, W]
        else:
            fi = frames_in
            yb = y_base
            cond = torch.cat([fi, yb], dim=1)  # [B, T_in+K, C, H, W]
        return cond
