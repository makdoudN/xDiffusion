"""Hold the checkpoint fixed while testing sampler resolution and stochasticity."""

from pathlib import Path

import torch

from xdiffusion.common import get_device, load_checkpoint, save_plot, write_json
from xdiffusion.data import DataConfig, load_data
from xdiffusion.quality import image_scores, load_encoder
from xdiffusion.sample import SamplingConfig, draw_samples


def main() -> None:
    torch.set_num_threads(4)
    device = get_device("auto")
    model, diffusion, checkpoint = load_checkpoint(Path("out/longer-images/mnist-seed0/ckpt.pt"), device)
    judge = load_encoder(Path("out/encoders/mnist/judge.pt"), device)
    real = load_data(DataConfig(task="mnist"), "val")[:512]
    root = Path("out/quality-ablation/samplers")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for sampler, steps in (("ddim", 50), ("ddim", 200), ("ddpm", 200)):
        samples = draw_samples(
            model, diffusion, tuple(checkpoint["shape"]), SamplingConfig(sampler=sampler, steps=steps)
        )
        result = {"sampler": sampler, "steps": steps, **image_scores(judge, real, samples)}
        results.append(result)
        save_plot(samples, root / f"{sampler}{steps}.png")
        print(result, flush=True)
    write_json(root / "results.json", results)


if __name__ == "__main__":
    main()
