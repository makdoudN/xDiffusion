"""Conditional flow matching along straight Gaussian-to-data paths.

x_t = (1-t) z + t x, velocity = x-z. Integrate t=0..1 at generation.
This is a flow-matching ODE, not a normalizing flow with a likelihood API.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class FlowConfig:
    steps: int = 25
    solver: Literal["euler", "heun"] = "heun"  # Heun uses two network evaluations per step

    def __post_init__(self):
        if self.steps < 1:
            raise ValueError("flow steps must be positive")


def flow_loss(
    model: nn.Module,
    clean: Float[Tensor, " batch *shape"],
    noise: Float[Tensor, " batch *shape"],
    time: Float[Tensor, " batch"],
) -> Float[Tensor, ""]:
    t = time.reshape(-1, *([1] * (clean.ndim - 1)))
    interpolated = (1 - t) * noise + t * clean
    return F.mse_loss(model(interpolated, time), clean - noise)


@torch.no_grad()
def integrate(
    model: nn.Module,
    x: Float[Tensor, " batch *shape"],
    cfg: FlowConfig,
    start: float = 0.0,
    end: float = 1.0,
) -> Float[Tensor, " batch *shape"]:
    dt = (end - start) / cfg.steps
    for step in range(cfg.steps):
        t = torch.full((len(x),), start + step * dt, device=x.device)
        velocity = model(x, t)
        proposed = x + dt * velocity
        if cfg.solver == "heun":
            next_velocity = model(proposed, t + dt)
            x = x + dt * (velocity + next_velocity) / 2
        elif cfg.solver == "euler":
            x = proposed
        else:
            raise ValueError(f"unknown flow solver: {cfg.solver}")
    return x
