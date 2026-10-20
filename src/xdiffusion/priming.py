"""A small Priming Model: fixed-noise reconstruction, then feature-space matching.

Implements the Laplacian-MMD alternative from Park et al., arXiv:2610.20331,
not the released ImageNet model or its OT/large-pretrained-encoder recipe.
The post-training reference negatives are an independent detached fake batch.
"""

import math
from dataclasses import replace

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from xdiffusion.models import ModelConfig, make_model
from xdiffusion.quality import FeatureNet


class Priming(nn.Module):
    def __init__(self, shape: tuple[int, ...], cfg: ModelConfig):
        super().__init__()
        self.image = len(shape) == 3
        self.network = make_model(shape, replace(cfg, identity_boundary=False))

    def forward(self, x: Float[Tensor, " batch *shape"]) -> Float[Tensor, " batch *shape"]:
        # Time is constant, so the generator is not conditioned on a noise schedule.
        output = self.network(x, torch.zeros(len(x), device=x.device))
        return output.tanh() if self.image else output


class FrozenFeatures(nn.Module):
    """Balance low-level spatial features and semantic features using training data.

    These weights never train with the generator. The evaluation judge is a
    separate model, initialized and trained independently.
    """

    def __init__(self, encoder: FeatureNet, calibration: Float[Tensor, " batch 1 28 28"]):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        with torch.no_grad():
            blocks = encoder.representations(calibration)
            scales = [
                b.std(dim=0).square().mean().sqrt().clamp_min(1e-6) * math.sqrt(b.shape[1]) for b in blocks
            ]
        self.register_buffer("scales", torch.stack(scales))

    def forward(self, images: Float[Tensor, " batch 1 28 28"]) -> Float[Tensor, " batch features"]:
        blocks = self.encoder.representations(images)
        return torch.cat([b / scale for b, scale in zip(blocks, self.scales, strict=True)], dim=-1)


def reconstruction_loss(
    model: nn.Module,
    clean: Float[Tensor, " batch *shape"],
    noise: Float[Tensor, " batch *shape"],
    level: float = 0.8,
) -> Float[Tensor, ""]:
    return F.mse_loss(model((1 - level) * clean + level * noise), clean)


def laplacian_mmd(
    online: Float[Tensor, " batch features"],
    real: Float[Tensor, " reference features"],
    negative: Float[Tensor, " other features"],
    scale: float = 1.0,
) -> Float[Tensor, ""]:
    """Attract to real features and repel from a stop-gradient generated batch.

    This is the paper's D(a,b)-D(a,sg[a]) surrogate. It has the gradient of
    half the population MMD², rather than being a numerical MMD² estimate.
    """
    if scale <= 0:
        raise ValueError("feature scale must be positive")
    bandwidth = scale * math.sqrt(online.shape[-1])
    # Explicit distance mode avoids cancellation at small distances.
    attraction = -torch.exp(
        -torch.cdist(online, real.detach(), compute_mode="donot_use_mm_for_euclid_dist") / bandwidth
    ).mean()
    repulsion = torch.exp(
        -torch.cdist(online, negative.detach(), compute_mode="donot_use_mm_for_euclid_dist") / bandwidth
    ).mean()
    return attraction + repulsion
