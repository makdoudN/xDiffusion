"""Compatibility flags for diffusion evaluation; all work runs through generate."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from xdiffusion.common import get_device
from xdiffusion.generate import GenerateConfig
from xdiffusion.generate import main as generate
from xdiffusion.run import load_run
from xdiffusion.sample import SamplingConfig


@dataclass
class EvaluateConfig:
    checkpoint: Path = Path("out/ring8/ckpt.pt")
    split: Literal["val", "test"] = "test"
    out_dir: Path = Path("out/evaluation")
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    device: str = "auto"
    threads: int = 4


def main(cfg: EvaluateConfig) -> None:
    _, run, _ = load_run(cfg.checkpoint, get_device(cfg.device))
    generate(
        GenerateConfig(
            checkpoint=cfg.checkpoint,
            out_dir=cfg.out_dir,
            split=cfg.split,
            device=cfg.device,
            threads=cfg.threads,
            samples=cfg.sampling.samples,
            batch_size=cfg.sampling.batch_size,
            seed=cfg.sampling.seed,
            sampler=cfg.sampling.sampler,
            sampling_steps=cfg.sampling.steps or run.diffusion_steps,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(EvaluateConfig))
