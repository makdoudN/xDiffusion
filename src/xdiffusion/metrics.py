"""Small, inspectable diagnostics. Pixel-space SWD is not a perceptual image score."""

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.diffusion import Diffusion


def ring8_scores(points: Float[Tensor, " sample 2"]) -> dict:
    """Separate mode coverage from a blurry ring, which projection distance can miss.

    Count points within 3 component standard deviations (0.24) of a center.
    A mode is covered if at least 1% of all generated points land near it.
    """
    angles = torch.arange(8) * torch.pi / 4
    centers = 1.5 * torch.stack([angles.cos(), angles.sin()], dim=-1)
    distance, nearest = torch.cdist(points.cpu().float(), centers).min(dim=-1)
    inside = distance < 0.24
    counts = nearest[inside].bincount(minlength=8)
    return {
        "within_mode_fraction": inside.float().mean().item(),
        "mode_counts": counts.tolist(),
        "covered_modes": int((counts >= max(1, len(points) // 100)).sum()),
    }


def sliced_wasserstein(
    x: Float[Tensor, " sample *shape"],
    y: Float[Tensor, " sample *shape"],
    projections: int = 128,
    seed: int = 0,
) -> float:
    """Mean 1D Wasserstein-1 over fixed unit projections; equal sample counts.

    Always computed on CPU in float32, with the same directions for every run.
    """
    if x.shape != y.shape or len(x) < 2 or projections < 1:
        raise ValueError("SWD needs matching shapes, at least two samples, and positive projections")
    x = rearrange(x.detach().cpu().float(), "n ... -> n (...)")
    y = rearrange(y.detach().cpu().float(), "n ... -> n (...)")
    directions = torch.randn(x.shape[1], projections, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)
    return ((x @ directions).sort(dim=0).values - (y @ directions).sort(dim=0).values).abs().mean().item()


@torch.no_grad()
def noise_mse(
    model: nn.Module,
    diffusion: Diffusion,
    data: Float[Tensor, " sample *shape"],
    batch_size: int,
    seed: int = 0,
) -> float:
    """Fixed data, times, and noise. No calls to the global training RNG."""
    device = diffusion.alpha_bar.device
    rng = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for clean in data.split(batch_size):
        clean = clean.to(device)
        t = torch.randint(1, diffusion.cfg.steps + 1, (len(clean),), device=device, generator=rng)
        noise = torch.randn(clean.shape, device=device, generator=rng)
        noisy = diffusion.add_noise(clean, t, noise)
        predicted = diffusion.predict_noise(model, noisy, t)
        total += (predicted - noise).square().mean().item() * len(clean)
    return total / len(data)
