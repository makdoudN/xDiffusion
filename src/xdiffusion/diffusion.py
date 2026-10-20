"""Epsilon or v prediction; DDPM and deterministic DDIM reverse processes.

Index 0 is clean data. Indices 1..T are noisy. Models receive t/T in (0, 1].
The leading batch axis is followed by either coordinates or channel/height/width.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

Sampler = Literal["ddim", "ddpm"]


@dataclass
class DiffusionConfig:
    steps: int = 200  # forward noise levels; DDPM uses all of them at sampling time
    prediction: Literal["epsilon", "v"] = "epsilon"  # v avoids dividing by tiny signal power at sampling


class Diffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        if cfg.steps < 2:
            raise ValueError("diffusion steps must be at least 2")
        self.cfg = cfg
        # Cosine cumulative signal power; cap beta to avoid a singular final step.
        time = torch.linspace(0, 1, cfg.steps + 1, dtype=torch.float64)
        signal = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
        beta = (1 - signal[1:] / signal[:-1]).clamp(max=0.999)
        alpha_bar = torch.cat([torch.ones(1), (1 - beta).cumprod(0)])
        self.register_buffer("alpha_bar", alpha_bar.float())

    def add_noise(
        self,
        clean: Float[Tensor, " batch *shape"],
        t: Int[Tensor, " batch"],
        noise: Float[Tensor, " batch *shape"],
    ) -> Float[Tensor, " batch *shape"]:
        a = self.alpha_bar[t].reshape(-1, *([1] * (clean.ndim - 1)))
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def loss(
        self,
        model: nn.Module,
        clean: Float[Tensor, " batch *shape"],
        t: Int[Tensor, " batch"],
        noise: Float[Tensor, " batch *shape"],
    ) -> Float[Tensor, ""]:
        noisy = self.add_noise(clean, t, noise)
        prediction = model(noisy, t.float() / self.cfg.steps)
        target = noise
        if self.cfg.prediction == "v":
            a = self.alpha_bar[t].reshape(-1, *([1] * (clean.ndim - 1)))
            target = a.sqrt() * noise - (1 - a).sqrt() * clean
        return F.mse_loss(prediction, target)

    def predict_clean(
        self, model: nn.Module, noisy: Float[Tensor, " batch *shape"], t: Int[Tensor, " batch"]
    ) -> Float[Tensor, " batch *shape"]:
        prediction = model(noisy, t.float() / self.cfg.steps)
        a = self.alpha_bar[t].reshape(-1, *([1] * (noisy.ndim - 1)))
        if self.cfg.prediction == "v":
            return a.sqrt() * noisy - (1 - a).sqrt() * prediction
        return (noisy - (1 - a).sqrt() * prediction) / a.sqrt()

    def predict_noise(
        self, model: nn.Module, noisy: Float[Tensor, " batch *shape"], t: Int[Tensor, " batch"]
    ) -> Float[Tensor, " batch *shape"]:
        prediction = model(noisy, t.float() / self.cfg.steps)
        if self.cfg.prediction == "v":
            a = self.alpha_bar[t].reshape(-1, *([1] * (noisy.ndim - 1)))
            return (1 - a).sqrt() * noisy + a.sqrt() * prediction
        return prediction

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        generator: torch.Generator,
        sampler: Sampler = "ddim",
        steps: int | None = None,
        clip: bool = False,
    ) -> Float[Tensor, " batch *shape"]:
        """Sample in eval mode. DDIM skips levels; DDPM uses posterior variance.

        Clip x0 only for images in [-1, 1]; point distributions are unbounded.
        Randomness comes entirely from `generator`, never the training RNG.
        """
        total = self.cfg.steps
        steps = total if steps is None else steps
        if not 1 <= steps <= total:
            raise ValueError("sampling steps must be between 1 and diffusion.steps")
        if sampler not in ("ddim", "ddpm"):
            raise ValueError(f"unknown sampler: {sampler}")
        if sampler == "ddpm" and steps != total:
            raise ValueError("DDPM requires all diffusion steps; use DDIM to skip levels")
        device = self.alpha_bar.device
        x = torch.randn(shape, device=device, generator=generator)
        # Python indices avoid a GPU-to-CPU sync in every reverse step.
        levels = torch.linspace(total, 0, steps + 1).round().long().tolist()
        for t, previous in zip(levels[:-1], levels[1:], strict=True):
            a, a_prev = self.alpha_bar[t], self.alpha_bar[previous]
            time = torch.full((shape[0],), t / total, device=device)
            prediction = model(x, time)
            clean = (
                a.sqrt() * x - (1 - a).sqrt() * prediction
                if self.cfg.prediction == "v"
                else (x - (1 - a).sqrt() * prediction) / a.sqrt()
            )
            if clip:
                clean = clean.clamp(-1, 1)
            if sampler == "ddim":
                # Recompute epsilon after clipping so x, x0 and epsilon agree.
                noise = (x - a.sqrt() * clean) / (1 - a).sqrt()
                x = a_prev.sqrt() * clean + (1 - a_prev).sqrt() * noise
            else:
                alpha = a / a_prev
                beta = 1 - alpha
                mean = (a_prev.sqrt() * beta * clean + alpha.sqrt() * (1 - a_prev) * x) / (1 - a)
                variance = beta * (1 - a_prev) / (1 - a)
                x = mean
                if previous > 0:  # the last transition is noiseless
                    x = x + variance.sqrt() * torch.randn(shape, device=device, generator=generator)
        return x
