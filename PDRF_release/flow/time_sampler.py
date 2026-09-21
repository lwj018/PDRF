"""Flow-time sampling for CRF training."""

import torch


class TimeSampler:
    """Sample the (discrete step-index) flow time for each batch element.

    t is a step index in [0, num_timesteps); the continuous flow time is
    t_cont = 1 - t / num_timesteps in (0, 1].
    """

    def __init__(self, use_discrete_timesteps=False):
        self.use_discrete_timesteps = use_discrete_timesteps

    def sample(self, x_start, num_timesteps):
        if self.use_discrete_timesteps:
            t = torch.randint(0, num_timesteps, (x_start.shape[0],), device=x_start.device)
        else:
            t = torch.rand((x_start.shape[0],), device=x_start.device) * num_timesteps
        return t
