"""Two denoisers with the same contract: (noisy data, normalized time) -> noise.

An MLP mixes coordinates; a two-resolution U-Net mixes neighboring pixels.
No base class, registry, attention, or architecture inheritance is needed.
"""

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width: int = 128  # MLP hidden width; U-Net base channels (use 32 for small images)
    depth: int = 3  # residual MLP blocks; U-Net has a fixed, readable topology
    time_dim: int = 32  # even number of sinusoidal time features

    def __post_init__(self):
        if self.width < 8 or self.width % 8:
            raise ValueError("width must be a positive multiple of 8")
        if self.depth < 1 or self.time_dim < 4 or self.time_dim % 2:
            raise ValueError("depth must be positive and time_dim must be even and >= 4")


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        frequencies = torch.exp(-math.log(10_000) * torch.arange(dim // 2) / (dim // 2 - 1))
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: Float[Tensor, " batch"]) -> Float[Tensor, " batch embedding"]:
        angles = t[:, None] * 1000 * self.frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, coordinates: int = 2):
        super().__init__()
        self.time = TimeEmbedding(cfg.time_dim)
        self.input = nn.Linear(coordinates + cfg.time_dim, cfg.width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.width), nn.SiLU())
            for _ in range(cfg.depth)
        )
        self.output = nn.Linear(cfg.width, coordinates)

    def forward(
        self, x: Float[Tensor, " batch coordinates"], t: Float[Tensor, " batch"]
    ) -> Float[Tensor, " batch coordinates"]:
        h = self.input(torch.cat([x, self.time(t)], dim=-1))
        for block in self.blocks:
            h = h + block(h)
        # At t=1, x is almost pure Gaussian noise: epsilon ≈ x. Enforce this
        # boundary rather than amplify a small prediction error by 1/sqrt(alpha_bar).
        # This matters for unbounded points, where clipping x0 would be incorrect.
        time = t[:, None]
        return time * x + (1 - time) * self.output(F.silu(h))


class ResBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.time = nn.Linear(time_dim, outgoing)
        self.norm2 = nn.GroupNorm(8, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(
        self, x: Float[Tensor, " batch incoming height width"], t: Float[Tensor, " batch embedding"]
    ) -> Float[Tensor, " batch outgoing height width"]:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + rearrange(self.time(F.silu(t)), "b c -> b c 1 1")
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class UNet(nn.Module):
    """Full resolution -> half resolution -> bottleneck -> full resolution + skip."""

    def __init__(self, cfg: ModelConfig, channels: int):
        super().__init__()
        w = cfg.width
        self.time = nn.Sequential(TimeEmbedding(cfg.time_dim), nn.Linear(cfg.time_dim, w), nn.SiLU())
        self.input = nn.Conv2d(channels, w, 3, padding=1)
        self.encoder = ResBlock(w, w, w)
        self.down = nn.Conv2d(w, 2 * w, 4, stride=2, padding=1)
        self.middle = ResBlock(2 * w, 2 * w, w)
        self.up = nn.Conv2d(2 * w, w, 3, padding=1)
        self.decoder = ResBlock(2 * w, w, w)
        self.output = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, channels, 3, padding=1))

    def forward(
        self, x: Float[Tensor, " batch channels height width"], t: Float[Tensor, " batch"]
    ) -> Float[Tensor, " batch channels height width"]:
        time = self.time(t)
        skip = self.encoder(self.input(x), time)
        h = self.middle(self.down(skip), time)
        h = self.up(F.interpolate(h, size=skip.shape[-2:], mode="nearest"))
        return self.output(self.decoder(torch.cat([h, skip], dim=1), time))


def make_model(shape: tuple[int, ...], cfg: ModelConfig) -> nn.Module:
    if len(shape) == 1:
        return MLP(cfg, coordinates=shape[0])
    if len(shape) == 3:
        return UNet(cfg, channels=shape[0])
    raise ValueError(f"expected coordinates or CHW images, got {shape}")
