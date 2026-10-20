"""Time-conditioned backbones: coordinates or pixels in, same-shaped field out.

An MLP mixes coordinates; U-Nets at two sizes mix pixels and image-wide context.
No base class, registry, attention, or architecture inheritance is needed.
"""

import math
from dataclasses import dataclass
from typing import Literal

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
    architecture: Literal["tiny", "unet"] = "unet"  # tiny reproduces the original two-resolution images
    identity_boundary: bool = True  # epsilon MLP only; disable for a velocity or clean-data field

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
        self.identity_boundary = cfg.identity_boundary
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
        output = self.output(F.silu(h))
        if self.identity_boundary:
            time = t[:, None]
            output = time * x + (1 - time) * output
        return output


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


class TinyUNet(nn.Module):
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


class UNet(nn.Module):
    """Three resolutions give the bottleneck context over the whole 28x28 image.

    Keep explicit skips and blocks: there is no recursive builder or attention
    machinery. The original two-resolution network remains available as `tiny`.
    """

    def __init__(self, cfg: ModelConfig, channels: int):
        super().__init__()
        w, time_dim = cfg.width, 4 * cfg.width
        self.time = nn.Sequential(
            TimeEmbedding(cfg.time_dim),
            nn.Linear(cfg.time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(channels, w, 3, padding=1)
        self.enc1 = ResBlock(w, w, time_dim)
        self.down1 = nn.Conv2d(w, 2 * w, 4, stride=2, padding=1)
        self.enc2 = ResBlock(2 * w, 2 * w, time_dim)
        self.down2 = nn.Conv2d(2 * w, 4 * w, 4, stride=2, padding=1)
        self.middle1 = ResBlock(4 * w, 4 * w, time_dim)
        self.middle2 = ResBlock(4 * w, 4 * w, time_dim)
        self.up2 = nn.Conv2d(4 * w, 2 * w, 3, padding=1)
        self.dec2 = ResBlock(4 * w, 2 * w, time_dim)
        self.up1 = nn.Conv2d(2 * w, w, 3, padding=1)
        self.dec1 = ResBlock(2 * w, w, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, channels, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, x: Float[Tensor, " batch channels height width"], t: Float[Tensor, " batch"]
    ) -> Float[Tensor, " batch channels height width"]:
        time = self.time(t)
        first = self.enc1(self.input(x), time)
        second = self.enc2(self.down1(first), time)
        h = self.middle2(self.middle1(self.down2(second), time), time)
        h = self.up2(F.interpolate(h, size=second.shape[-2:], mode="nearest"))
        h = self.dec2(torch.cat([h, second], dim=1), time)
        h = self.up1(F.interpolate(h, size=first.shape[-2:], mode="nearest"))
        return self.output(self.dec1(torch.cat([h, first], dim=1), time))


def make_model(shape: tuple[int, ...], cfg: ModelConfig) -> nn.Module:
    if len(shape) == 1:
        return MLP(cfg, coordinates=shape[0])
    if len(shape) == 3:
        network = TinyUNet if cfg.architecture == "tiny" else UNet
        return network(cfg, channels=shape[0])
    raise ValueError(f"expected coordinates or CHW images, got {shape}")
