"""Sample or evaluate any family checkpoint, including real VAE reconstructions."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from einops import rearrange

from xdiffusion.common import get_device, save_plot, write_json
from xdiffusion.data import load_data
from xdiffusion.diffusion import Diffusion, DiffusionConfig, Sampler
from xdiffusion.metrics import noise_mse
from xdiffusion.quality import accuracy, load_encoder, load_labels
from xdiffusion.run import generate, load_run, score_samples
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
    evaluate: bool = True  # False writes samples without loading a reference dataset
    sampler: Sampler | None = None  # diffusion-only override; otherwise use the checkpoint
    sampling_steps: int | None = None
    threads: int = 4

    def __post_init__(self):
        if self.samples < 2 or min(self.batch_size, self.threads) < 1:
            raise ValueError("samples must be >= 2; batch size and threads must be positive")


@torch.no_grad()
def main(cfg: GenerateConfig) -> None:
    torch.set_num_threads(cfg.threads)
    device = get_device(cfg.device)
    model, experiment, shape = load_run(cfg.checkpoint, device, require_data=cfg.evaluate)
    experiment = replace(
        experiment,
        samples=cfg.samples,
        sample_batch_size=cfg.batch_size,
        sample_seed=cfg.seed,
        sampler=cfg.sampler or experiment.sampler,
        sampling_steps=cfg.sampling_steps if cfg.sampling_steps is not None else experiment.sampling_steps,
    )
    if experiment.family != "diffusion" and (cfg.sampler is not None or cfg.sampling_steps is not None):
        raise ValueError("sampler overrides apply only to diffusion checkpoints")
    real = load_data(experiment.data, cfg.split) if cfg.evaluate else None
    if real is not None and len(real) < 2 * cfg.samples:
        raise ValueError("evaluation needs two reference sets as large as the sample set")
    samples = generate(model, experiment, shape)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, cfg.out_dir / "samples.pt")
    save_plot(samples, cfg.out_dir / "samples.png", real[: cfg.samples] if real is not None else None)
    if real is None:
        return
    scores = score_samples(samples, real, experiment, device)
    scores.update(
        split=cfg.split,
        checkpoint=str(cfg.checkpoint),
        family=experiment.family,
        sample_count=cfg.samples,
        sampling_seed=cfg.seed,
        sampling_batch_size=cfg.batch_size,
    )
    if experiment.family == "diffusion":
        diffusion = Diffusion(DiffusionConfig(experiment.diffusion_steps, experiment.prediction)).to(device)
        scores.update(
            sampler=experiment.sampler,
            sampling_steps=experiment.sampling_steps,
            noise_mse=noise_mse(model, diffusion, real[: cfg.samples], cfg.batch_size, cfg.seed),
        )
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
        if experiment.encoder_dir is not None and experiment.data.task in ("mnist", "fashion"):
            judge = load_encoder(
                experiment.encoder_dir / "judge.pt", device, experiment.data.task, experiment.data.seed
            )
            labels = load_labels(experiment.data, cfg.split)[: len(originals)]
            scores["original_label_accuracy"] = accuracy(judge, originals, labels)
            scores["reconstruction_label_accuracy"] = accuracy(judge, reconstructions, labels)
        torch.save(
            {"originals": originals, "reconstructions": reconstructions},
            cfg.out_dir / "reconstructions.pt",
        )
    write_json(cfg.out_dir / "evaluation.json", scores)
    print(scores)


if __name__ == "__main__":
    main(tyro.cli(GenerateConfig))
