"""CRF - Conditional Rectified Flow with the MASD data-space parameterization.

MASD (Manifold-Anchored Spatiotemporal Dynamics, paper Sec. 3.1):
the network does NOT regress a velocity; it acts as a manifold projector
that predicts the clean future sequence

    X1hat = f_theta(Z_t, t, C)                                  (Eq. 4)

and the velocity field is induced analytically, pointing toward the
predicted physical support:

    v_hat = (X1hat - Z_t) / tbar,   tbar = max(1 - t, eps)        (Eq. 5)

Training objective (paper Eqs. 7, 13, 14):

    L_total = L_flow + lambda_phy * L_phy
    L_flow  = E || v_hat - (X1 - Z_t) / tbar ||^2
    L_phy   = E || v_hat - v_SL ||^2,   v_SL = (X_SL - Z_t) / tbar

where X_SL is the Semi-Lagrangian advection prior used as a soft teacher
(physics-guided velocity regularizer, paper Sec. 3.2).

Inference (paper Sec. 4.1): a fixed 5-step Euler ODE in data space.
"""

import torch

from .time_sampler import TimeSampler


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class CRFScheduler:
    """Flow-matching scheduler for the data-space (MASD) parameterization."""

    def __init__(self, num_timesteps=1000, num_sampling_steps=5,
                 use_discrete_timesteps=False, lambda_phy=0.1):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps
        self.lambda_phy = lambda_phy
        self.time_sampler = TimeSampler(use_discrete_timesteps=use_discrete_timesteps)

    def add_noise(self, original_samples, noise, timesteps):
        """Linear path: Z_t = t_cont * X1 + (1 - t_cont) * noise.

        timesteps: step index in [0, num_timesteps); t_cont = 1 - t / num_timesteps.
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints
        timepoints = timepoints.view(-1, 1, 1, 1)
        return timepoints * original_samples + (1 - timepoints) * noise

    def training_losses(self, model, x_start, condition, x_sl=None, noise=None, t=None):
        """L_total = L_flow + lambda_phy * L_phy  (Eqs. 7, 13, 14).

        model:      f_theta, predicts the clean future sequence [B, T_out, H, W]
        x_start:    target sequence X1, [B, T_out, H, W]
        condition:  historical context C, [B, T_in, H, W]
        x_sl:       Semi-Lagrangian prior X_SL, [B, T_out, H, W]
        """
        if t is None:
            t = self.time_sampler.sample(x_start, self.num_timesteps)
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        # forward noising
        x_t = self.add_noise(x_start, noise, t)

        # MASD: the network directly predicts the clean future sequence
        x_pred = model(x_t, t, condition)

        # induced velocity: v = (x - x_t) / tbar
        t_cont = 1.0 - t.float() / float(self.num_timesteps)
        one_minus_t = (1.0 - t_cont).clamp(min=1e-3).view(-1, 1, 1, 1)

        v_pred = (x_pred - x_t) / one_minus_t
        v_target = (x_start - x_t) / one_minus_t
        loss_flow = mean_flat((v_pred - v_target).pow(2)).sum()

        if x_sl is not None:
            v_sl = (x_sl - x_t) / one_minus_t
            loss_phy = mean_flat((v_pred - v_sl).pow(2)).sum()
        else:
            loss_phy = torch.zeros((), device=x_pred.device)

        terms = {"loss_flow": loss_flow, "loss_phy": loss_phy}
        terms["loss"] = loss_flow + self.lambda_phy * loss_phy
        return terms

    def sample(self, model, z, condition, device, progress=True):
        """Fixed 5-step Euler ODE integration in data space (paper Sec. 4.1)."""
        from tqdm import tqdm

        B = z.shape[0]
        num_steps = self.num_sampling_steps
        dt_scalar = 1.0 / float(num_steps)
        expand_shape = (B,) + (1,) * (z.ndim - 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i in progress_wrap(range(num_steps)):
            # continuous flow time t_cont in [0, 1)
            t_cont = float(i) / float(num_steps)
            # step index expected by the network
            step_value = (1.0 - t_cont) * float(self.num_timesteps)
            step_value = max(0.0, min(float(self.num_timesteps - 1), step_value))
            t_model = torch.full((B,), step_value, device=device)
            if self.use_discrete_timesteps:
                t_model = t_model.round().long()

            # 1) predict the clean future sequence
            x_pred = model(z, t_model, condition)

            # 2) induced velocity with the same stabilized scaling as training
            one_minus_t = torch.full(
                expand_shape, max(1.0 - t_cont, 1e-3), device=device, dtype=z.dtype
            )
            v_pred = (x_pred - z) / one_minus_t

            # 3) ODE step: z_{t+dt} = z_t + v(x_t, t) * dt
            dt = torch.full(expand_shape, dt_scalar, device=device, dtype=z.dtype)
            z = z + v_pred * dt

        return z


class CRF:
    """Conditional Rectified Flow: scheduler + data-space ODE sampler."""

    def __init__(self, num_sampling_steps=5, num_timesteps=1000,
                 use_discrete_timesteps=False, lambda_phy=0.1):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.lambda_phy = lambda_phy
        self.scheduler = CRFScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            lambda_phy=lambda_phy,
        )

    def training_losses(self, model, x_start, condition, x_sl=None, noise=None, t=None):
        return self.scheduler.training_losses(
            model, x_start, condition, x_sl, noise=noise, t=t
        )

    def sample(self, model, z, condition, device, progress=True):
        return self.scheduler.sample(model, z, condition, device, progress=progress)
