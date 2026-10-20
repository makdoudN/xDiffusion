"""Sample or evaluate any family checkpoint, including real VAE reconstructions."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from einops import rearrange

from xdiffusion.common import get_device, save_plot, write_json
from xdiffusion.data import load_data
from xdiffusion.experiment import generate, load_experiment, score_samples
from xdiffusion.vae import VAE


@dataclass
class GenerateConfig:
    checkpoint: Path
    out_dir: Path = Path("out/generated")
    samples: int = 1024
    batch_size: int = 128
    seed: int = 0
    split: Literal["val", "test"] = "test"
    device: str = "auto"


@torch.no_grad()
def main(cfg: GenerateConfig) -> None:
    torch.set_num_threads(4)
    device = get_device(cfg.device)
    model, experiment, shape = load_experiment(cfg.checkpoint, device)
    experiment = replace(
        experiment, samples=cfg.samples, sample_batch_size=cfg.batch_size, sample_seed=cfg.seed
    )
    samples = generate(model, experiment, shape)
    real = load_data(experiment.data, cfg.split)
    scores = score_samples(samples, real, experiment, device)
    scores.update(split=cfg.split, checkpoint=str(cfg.checkpoint), family=experiment.family)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, cfg.out_dir / "samples.pt")
    save_plot(samples, cfg.out_dir / "samples.png", real[: cfg.samples])
    if isinstance(model, VAE):
        originals = real[: cfg.samples]
        reconstructions = torch.cat(
            [model.reconstruct(batch.to(device)).cpu() for batch in originals.split(cfg.batch_size)]
        )
        pairs = rearrange(
            torch.stack([originals[:32], reconstructions[:32]], dim=1), "n pair ... -> (n pair) ..."
        )
        save_plot(pairs, cfg.out_dir / "reconstructions.png")
        scores["reconstruction_count"] = len(originals)
        scores["reconstruction_mse"] = (reconstructions - originals).square().mean().item()
        if len(shape) == 3:
            scores["reconstruction_mse_01"] = scores["reconstruction_mse"] / 4
        torch.save(
            {"originals": originals, "reconstructions": reconstructions},
            cfg.out_dir / "reconstructions.pt",
        )
    write_json(cfg.out_dir / "evaluation.json", scores)
    print(scores)


if __name__ == "__main__":
    main(tyro.cli(GenerateConfig))
