"""Generate points or an image grid from a saved checkpoint."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.common import get_device, load_checkpoint, save_plot
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
    rng = torch.Generator(device=diffusion.alpha_bar.device).manual_seed(cfg.seed)
    batches = []
    for start in range(0, cfg.samples, cfg.batch_size):
        n = min(cfg.batch_size, cfg.samples - start)
        batch = diffusion.sample(model, (n, *shape), rng, cfg.sampler, cfg.steps, clip=len(shape) == 3)
        batches.append(batch.cpu())
    return torch.cat(batches)


@dataclass
class SampleConfig:
    checkpoint: Path = Path("out/ring8/ckpt.pt")
    out_dir: Path = Path("out/samples")
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    device: str = "auto"
    threads: int = 4  # small models can be slower with many CPU threads


def main(cfg: SampleConfig) -> None:
    torch.set_num_threads(cfg.threads)
    model, diffusion, checkpoint = load_checkpoint(cfg.checkpoint, get_device(cfg.device))
    samples = draw_samples(model, diffusion, tuple(checkpoint["shape"]), cfg.sampling)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, cfg.out_dir / "samples.pt")
    save_plot(samples, cfg.out_dir / "samples.png")
    print(f"saved {len(samples)} samples to {cfg.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(SampleConfig))
