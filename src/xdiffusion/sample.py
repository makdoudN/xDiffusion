"""Compatibility flags for diffusion sampling; new code uses generate directly."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.common import get_device
from xdiffusion.diffusion import Diffusion, Sampler


@dataclass
class SamplingConfig:
    samples: int = 512
    batch_size: int = 128  # cap memory independently of sample count
    sampler: Sampler = "ddim"
    steps: int | None = 50  # DDPM requires --sampling.steps None (all forward levels)
    seed: int = 0

    def __post_init__(self):
        if self.samples < 2 or self.batch_size < 1:
            raise ValueError("samples must be >= 2 and batch_size must be positive")


@torch.no_grad()
def draw_samples(
    model: nn.Module, diffusion: Diffusion, shape: tuple[int, ...], cfg: SamplingConfig
) -> Float[Tensor, " sample *shape"]:
    from xdiffusion.run import TrainConfig, generate

    return generate(
        model,
        TrainConfig(
            diffusion_steps=diffusion.cfg.steps,
            prediction=diffusion.cfg.prediction,
            samples=cfg.samples,
            sample_batch_size=cfg.batch_size,
            sample_seed=cfg.seed,
            sampler=cfg.sampler,
            sampling_steps=cfg.steps or diffusion.cfg.steps,
        ),
        shape,
    )


@dataclass
class SampleConfig:
    checkpoint: Path = Path("out/ring8/ckpt.pt")
    out_dir: Path = Path("out/samples")
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    device: str = "auto"
    threads: int = 4  # small models can be slower with many CPU threads


def main(cfg: SampleConfig) -> None:
    from xdiffusion.generate import GenerateConfig
    from xdiffusion.generate import main as generate
    from xdiffusion.run import load_run

    _, run, _ = load_run(cfg.checkpoint, get_device(cfg.device))
    generate(
        GenerateConfig(
            checkpoint=cfg.checkpoint,
            out_dir=cfg.out_dir,
            device=cfg.device,
            threads=cfg.threads,
            samples=cfg.sampling.samples,
            batch_size=cfg.sampling.batch_size,
            seed=cfg.sampling.seed,
            sampler=cfg.sampling.sampler,
            sampling_steps=cfg.sampling.steps or run.diffusion_steps,
            evaluate=False,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(SampleConfig))
