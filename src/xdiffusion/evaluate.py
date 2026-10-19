"""Evaluate a checkpoint with fixed noise and independent reference samples."""

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.common import get_device, load_checkpoint, save_plot, synchronize, write_json
from xdiffusion.data import DataConfig, load_data
from xdiffusion.diffusion import Diffusion
from xdiffusion.metrics import noise_mse, sliced_wasserstein
from xdiffusion.sample import SamplingConfig, draw_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    diffusion: Diffusion,
    data: Float[Tensor, " dataset *shape"],
    cfg: SamplingConfig,
) -> tuple[dict, Float[Tensor, " sample *shape"]]:
    n = cfg.samples
    if 2 * n > len(data):
        raise ValueError(f"evaluation needs 2 * samples <= split size ({len(data)}), got {n}")
    model.eval()
    shape = tuple(data.shape[1:])
    device = diffusion.alpha_bar.device
    mse = noise_mse(model, diffusion, data[:n], cfg.batch_size, cfg.seed)
    draw_samples(model, diffusion, shape, replace(cfg, samples=2))  # kernel warmup, excluded from timing
    synchronize(device)
    start = time.perf_counter()
    samples = draw_samples(model, diffusion, shape, cfg)
    synchronize(device)
    seconds = time.perf_counter() - start
    gaussian = torch.randn(data[:n].shape, generator=torch.Generator().manual_seed(cfg.seed))
    scores = {
        "noise_mse": mse,
        "swd": sliced_wasserstein(data[:n], samples),
        "reference_swd": sliced_wasserstein(data[:n], data[n : 2 * n]),
        "gaussian_swd": sliced_wasserstein(data[:n], gaussian),
        "sample_count": n,
        "sampler": cfg.sampler,
        "sampling_steps": cfg.steps if cfg.steps is not None else diffusion.cfg.steps,
        "sampling_seed": cfg.seed,
        "sampling_batch_size": cfg.batch_size,
        "sample_s": seconds,
        "samples_per_s": n / seconds,
    }
    return scores, samples


@dataclass
class EvaluateConfig:
    checkpoint: Path = Path("out/ring8/ckpt.pt")
    split: Literal["val", "test"] = "test"
    out_dir: Path = Path("out/evaluation")
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    device: str = "auto"
    threads: int = 4


def main(cfg: EvaluateConfig) -> None:
    torch.set_num_threads(cfg.threads)
    model, diffusion, checkpoint = load_checkpoint(cfg.checkpoint, get_device(cfg.device))
    data_cfg = DataConfig(**{**checkpoint["data_config"], "root": Path(checkpoint["data_config"]["root"])})
    data = load_data(data_cfg, cfg.split)
    scores, samples = evaluate(model, diffusion, data, cfg.sampling)
    scores.update(split=cfg.split, checkpoint=str(cfg.checkpoint), checkpoint_step=checkpoint["step"])
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.out_dir / "evaluation.json", scores)
    torch.save(samples, cfg.out_dir / "samples.pt")
    save_plot(samples, cfg.out_dir / "samples.png", data[: cfg.sampling.samples])
    print(f"{cfg.split} noise MSE {scores['noise_mse']:.4f} | SWD {scores['swd']:.4f} -> {cfg.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(EvaluateConfig))
