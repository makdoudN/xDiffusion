"""Finite, independently seeded synthetic splits or official torchvision datasets.

Run `python -m xdiffusion.data --task mnist` once to download an image dataset.
Training never downloads implicitly. The official test split stays untouched.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

Task = Literal["gaussian", "ring8", "moons", "checkerboard", "shapes", "mnist", "fashion", "cifar10"]
Split = Literal["train", "val", "test"]
POINT_TASKS = ("gaussian", "ring8", "moons", "checkerboard")
IMAGE_TASKS = ("mnist", "fashion", "cifar10")


@dataclass
class DataConfig:
    task: Task = "ring8"
    root: Path = Path("data")  # torchvision download cache
    train_size: int = 20_000  # synthetic only; image datasets use their full training partition
    val_size: int = 4096  # synthetic only; real images reserve 5,000 training examples
    seed: int = 1234  # fixes data independently of model/training seeds


def synthetic(task: Task, count: int, seed: int) -> Float[Tensor, " sample *shape"]:
    """Points have natural coordinates; images have pixels in [-1, 1]."""
    if count < 1:
        raise ValueError("dataset size must be positive")
    rng = torch.Generator().manual_seed(seed)
    if task == "gaussian":
        return torch.randn(count, 2, generator=rng) * torch.tensor([0.35, 0.15]) + torch.tensor([0.6, -0.4])
    if task == "ring8":
        angle = torch.randint(8, (count,), generator=rng) * (torch.pi / 4)
        centers = 1.5 * torch.stack([angle.cos(), angle.sin()], dim=-1)
        return centers + 0.08 * torch.randn(count, 2, generator=rng)
    if task == "moons":
        angle = torch.rand(count, generator=rng) * torch.pi
        side = torch.randint(2, (count,), generator=rng).bool()
        x = torch.where(side, 1 - angle.cos(), angle.cos())
        y = torch.where(side, 0.5 - angle.sin(), angle.sin())
        return (torch.stack([x, y], dim=-1) - torch.tensor([0.5, 0.25])) / 1.5 + (
            0.035 * torch.randn(count, 2, generator=rng)
        )
    if task == "checkerboard":
        row = torch.randint(4, (count,), generator=rng)
        col = 2 * torch.randint(2, (count,), generator=rng) + row % 2
        cells = torch.stack([col, row], dim=-1)
        return (cells + torch.rand(count, 2, generator=rng)) / 2 - 1
    if task == "shapes":
        # A translated, resized circle or square on a 16x16 canvas.
        axis = torch.linspace(-1, 1, 16)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        centers = torch.rand(count, 2, generator=rng) * 0.9 - 0.45
        radius = torch.rand(count, 1, 1, generator=rng) * 0.25 + 0.15
        dx = x[None] - centers[:, 0, None, None]
        dy = y[None] - centers[:, 1, None, None]
        square = torch.randint(2, (count, 1, 1), generator=rng).bool()
        mask = torch.where(
            square, torch.maximum(dx.abs(), dy.abs()) < radius, dx.square() + dy.square() < radius**2
        )
        brightness = 0.5 + torch.rand(count, 1, 1, generator=rng) * 0.5
        return rearrange(mask.float() * brightness * 2 - 1, "n h w -> n 1 h w")
    raise ValueError(f"not a synthetic task: {task}")


def load_data(
    cfg: DataConfig, split: Split = "train", download: bool = False
) -> Float[Tensor, " sample *shape"]:
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split}")
    if cfg.task not in IMAGE_TASKS:
        seed = cfg.seed + {"train": 0, "val": 1, "test": 2}[split]
        return synthetic(cfg.task, cfg.train_size if split == "train" else cfg.val_size, seed)

    # Keep torchvision out of the synthetic path.
    from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

    dataset_type = {"mnist": MNIST, "fashion": FashionMNIST, "cifar10": CIFAR10}[cfg.task]
    dataset = dataset_type(str(cfg.root), train=split != "test", download=download)
    pixels = torch.as_tensor(dataset.data)
    if cfg.task == "cifar10":
        pixels = rearrange(pixels, "n h w c -> n c h w")
    else:
        pixels = rearrange(pixels, "n h w -> n 1 h w")
    if split != "test":
        # Same permutation for train and val, regardless of training seed.
        order = torch.randperm(len(pixels), generator=torch.Generator().manual_seed(cfg.seed))
        pixels = pixels[order[5000:] if split == "train" else order[:5000]]
    return pixels.float() / 127.5 - 1


def main(cfg: DataConfig) -> None:
    for split in ("train", "val", "test"):
        data = load_data(cfg, split, download=True)
        print(f"{cfg.task:12} {split:5} {tuple(data.shape)} range [{data.min():.2f}, {data.max():.2f}]")


if __name__ == "__main__":
    main(tyro.cli(DataConfig))
