"""Keep unconditional samples, VAE compression, and denoising panels separate."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from einops import rearrange

from xdiffusion.common import get_device, save_plot, write_json
from xdiffusion.data import load_data
from xdiffusion.diffusion import Diffusion, DiffusionConfig
from xdiffusion.experiment import load_experiment
from xdiffusion.vae import VAE


@dataclass
class Config:
    run: Path
    out_dir: Path
    device: str = "auto"


@torch.no_grad()
def main(cfg: Config) -> None:
    torch.set_num_threads(4)
    device = get_device(cfg.device)
    model, experiment, _ = load_experiment(cfg.run / "ckpt.pt", device)
    clean = load_data(experiment.data, "test")[:16].to(device)
    rng = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(clean.shape, generator=rng, device=device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, VAE):
        reconstruction = model.reconstruct(clean)
        panels = [clean, reconstruction]
        labels = ["original", "posterior-mean reconstruction"]
    elif experiment.family == "diffusion":
        diffusion = Diffusion(DiffusionConfig(experiment.diffusion_steps, experiment.prediction)).to(device)
        level = experiment.diffusion_steps // 4
        t = torch.full((len(clean),), level, device=device, dtype=torch.long)
        noisy = diffusion.add_noise(clean, t, noise)
        reconstruction = diffusion.predict_clean(model, noisy, t).clamp(-1, 1)
        panels = [clean, noisy.clamp(-1, 1), reconstruction]
        labels = ["original", "noisy at t/T=0.25", "single denoising prediction"]
    elif experiment.family == "priming":
        primed, _, _ = load_experiment(cfg.run / "primed.pt", device)
        noisy = (1 - experiment.priming_level) * clean + experiment.priming_level * noise
        reconstruction = model(noisy)
        panels = [clean, noisy.clamp(-1, 1), primed(noisy), reconstruction]
        labels = ["original", "corrupted at s=0.8", "stage-1 denoising", "after distributional post-training"]
    else:
        raise ValueError("GAN has no encoder; flow cycle inversion is not a reconstruction-learning score")
    # At most 16 * 4 images: save_plot displays all panels, never crops the final stage.
    images = rearrange(torch.stack(panels, dim=1), "n panel ... -> (n panel) ...")
    save_plot(images.cpu(), cfg.out_dir / "reconstruction.png")
    data = {"original": clean.cpu(), "reconstruction": reconstruction.cpu()}
    torch.save(data, cfg.out_dir / "reconstruction.pt")
    write_json(
        cfg.out_dir / "reconstruction.json",
        {
            "family": experiment.family,
            "split": "test",
            "panel_order": labels,
            "mse_01": ((reconstruction - clean) / 2).square().mean().item(),
            "meaning": "VAE compression"
            if experiment.family == "vae"
            else "denoising with informative corrupted input; not unconditional generation",
        },
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
