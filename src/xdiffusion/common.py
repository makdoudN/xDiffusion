"""Only run plumbing: device selection, serialization, checkpoints, and plots."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.diffusion import Diffusion, DiffusionConfig


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name != "cpu" and not name.startswith("cuda"):
        raise ValueError("supported devices: auto, cpu, cuda, cuda:N")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(asdict(value) if is_dataclass(value) else value, indent=2, default=str) + "\n")


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, Diffusion, dict]:
    from xdiffusion.run import load_run

    model, cfg, shape = load_run(path, device)
    if cfg.family != "diffusion":
        raise ValueError("this compatibility loader is diffusion-only; use run.load_run")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return model, Diffusion(DiffusionConfig(cfg.diffusion_steps, cfg.prediction)).to(device), checkpoint


def save_plot(
    samples: Float[Tensor, " sample *shape"],
    path: Path,
    reference: Float[Tensor, " reference *shape"] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = samples.detach().cpu()
    if samples.ndim == 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        if reference is not None:
            ax.scatter(*reference.cpu().T, s=4, alpha=0.2, label="reference")
        ax.scatter(*samples.T, s=4, alpha=0.4, label="generated")
        ax.set_aspect("equal")
        ax.legend()
    else:
        images = (samples[:64].clamp(-1, 1) + 1) / 2
        columns = min(8, len(images))
        rows = (len(images) + columns - 1) // columns
        # Pad the last row, then put tiles next to one another using named axes.
        padded = images.new_zeros(rows * columns, *images.shape[1:])
        padded[: len(images)] = images
        grid = rearrange(padded, "(r c) ch h w -> (r h) (c w) ch", r=rows, c=columns)
        aspect = rows * images.shape[-2] / (columns * images.shape[-1])
        fig, ax = plt.subplots(figsize=(6, max(1.0, 6 * aspect)))
        pixels = grid.numpy()
        ax.imshow(pixels.squeeze(-1) if pixels.shape[-1] == 1 else pixels, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
