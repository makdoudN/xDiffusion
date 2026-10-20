"""Deterministic quality regression check for a saved MNIST sample batch.

This is a diagnostic gate, always paired with visual inspection. It deliberately
fails for the original README's undertrained samples; it is not a universal
perceptual metric or a substitute for a human looking at the output.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from xdiffusion.common import get_device, write_json
from xdiffusion.data import DataConfig, load_data
from xdiffusion.quality import image_scores, load_encoder


@dataclass
class Config:
    samples: Path = Path("out/longer-images/mnist-seed0/samples.pt")
    judge: Path = Path("out/encoders/mnist/judge.pt")
    report: Path = Path("out/original-mnist-quality.json")
    minimum_confident_fraction: float = 0.9
    limit: int = 512  # fixed count for before/after comparisons
    device: str = "auto"


def main(cfg: Config) -> None:
    torch.set_num_threads(4)
    samples = torch.load(cfg.samples, weights_only=True)[: cfg.limit]
    real = load_data(DataConfig(task="mnist"), "val")[: len(samples)]
    judge = load_encoder(cfg.judge, get_device(cfg.device))
    scores = image_scores(judge, real, samples)
    scores["real_confident_fraction"] = image_scores(judge, real, real)["confident_fraction"]
    write_json(cfg.report, scores)
    print(scores)
    assert scores["confident_fraction"] >= cfg.minimum_confident_fraction, (
        f"only {scores['confident_fraction']:.1%} confident images; "
        f"target >= {cfg.minimum_confident_fraction:.1%}"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
