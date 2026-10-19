"""One explicit training loop for every dataset and both denoisers."""

import math
import platform
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import tyro

from xdiffusion.common import get_device, load_checkpoint, save_plot, synchronize, write_json
from xdiffusion.data import DataConfig, load_data
from xdiffusion.diffusion import Diffusion, DiffusionConfig
from xdiffusion.evaluate import evaluate
from xdiffusion.metrics import noise_mse
from xdiffusion.models import ModelConfig, make_model
from xdiffusion.sample import SamplingConfig


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)  # final validation sampling
    out_dir: Path = Path("out/ring8")  # must not already contain a run
    steps: int = 3000
    batch_size: int = 256
    lr: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.995  # exponential moving average; 0 uses current weights
    eval_interval: int = 500
    log_interval: int = 100
    seed: int = 0  # model initialization and training randomness; data seed is separate
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    threads: int = 4

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.eval_interval, self.log_interval, self.threads) < 1:
            raise ValueError("steps, batch sizes, intervals and threads must be positive")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0 or self.warmup_steps < 0:
            raise ValueError("invalid optimizer settings")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")


def learning_rate(step: int, cfg: TrainConfig) -> float:
    warmup = min(cfg.warmup_steps, cfg.steps)
    if step <= warmup:
        return cfg.lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main(cfg: TrainConfig) -> dict:
    if (cfg.out_dir / "config.json").exists():
        raise FileExistsError(f"run already exists: {cfg.out_dir}; choose a new --out-dir")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    train_data, val_data = load_data(cfg.data), load_data(cfg.data, "val")
    if 2 * cfg.sampling.samples > len(val_data):
        raise ValueError("validation split must contain at least 2 * sampling.samples")
    sampling_steps = cfg.sampling.steps if cfg.sampling.steps is not None else cfg.diffusion.steps
    if not 1 <= sampling_steps <= cfg.diffusion.steps:
        raise ValueError("sampling.steps must be between 1 and diffusion.steps")
    if cfg.sampling.sampler == "ddpm" and sampling_steps != cfg.diffusion.steps:
        raise ValueError("DDPM needs --sampling.steps None or diffusion.steps")
    shape = tuple(train_data.shape[1:])
    model = make_model(shape, cfg.model).to(device)
    ema = deepcopy(model).requires_grad_(False).eval()
    diffusion = Diffusion(cfg.diffusion).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Data selection stays on CPU; noise and time are generated on the training device.
    data_rng = torch.Generator().manual_seed(cfg.seed)
    noise_rng = torch.Generator(device=device).manual_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.out_dir / "config.json", cfg)
    params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.data.task} | {device} | {params:,} parameters | shape {shape}", flush=True)
    initial_mse = noise_mse(
        ema, diffusion, val_data[: cfg.sampling.samples], cfg.sampling.batch_size, cfg.sampling.seed
    )
    history = [{"step": 0, "val_noise_mse": initial_mse, "train_s": 0.0}]
    best_mse, best_step = math.inf, 0
    train_s = 0.0
    synchronize(device)
    start = segment_start = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        indices = torch.randint(len(train_data), (cfg.batch_size,), generator=data_rng)
        clean = train_data[indices].to(device)
        t = torch.randint(1, cfg.diffusion.steps + 1, (len(clean),), device=device, generator=noise_rng)
        noise = torch.randn(clean.shape, device=device, generator=noise_rng)
        loss = diffusion.loss(model, clean, t, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            # Early averaging should follow the model rather than preserve its random initialization.
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            for averaged, current in zip(ema.parameters(), model.parameters(), strict=True):
                averaged.lerp_(current, 1 - decay)

        if step % cfg.log_interval == 0:
            print(f"step {step:6} | train MSE {loss.item():.4f}", flush=True)
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            synchronize(device)
            train_s += time.perf_counter() - segment_start
            val_mse = noise_mse(
                ema, diffusion, val_data[: cfg.sampling.samples], cfg.sampling.batch_size, cfg.sampling.seed
            )
            history.append({"step": step, "val_noise_mse": val_mse, "train_s": train_s})
            write_json(cfg.out_dir / "history.json", history)
            if val_mse < best_mse:
                best_mse, best_step = val_mse, step
                torch.save(
                    {
                        "model": ema.state_dict(),
                        "model_config": asdict(cfg.model),
                        "diffusion_config": asdict(cfg.diffusion),
                        "data_config": {**asdict(cfg.data), "root": str(cfg.data.root)},
                        "shape": shape,
                        "step": step,
                        "val_noise_mse": val_mse,
                    },
                    cfg.out_dir / "ckpt.pt",
                )
            print(f"step {step:6} | val MSE {val_mse:.4f} | best {best_mse:.4f}", flush=True)
            synchronize(device)
            segment_start = time.perf_counter()

    best_model, diffusion, _ = load_checkpoint(cfg.out_dir / "ckpt.pt", device)
    scores, samples = evaluate(best_model, diffusion, val_data, cfg.sampling)
    torch.save(samples, cfg.out_dir / "samples.pt")
    save_plot(samples, cfg.out_dir / "samples.png", val_data[: cfg.sampling.samples])
    summary = {
        "task": cfg.data.task,
        "seed": cfg.seed,
        "data_seed": cfg.data.seed,
        "split": "val",
        "params": params,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "train_examples": cfg.steps * cfg.batch_size,
        "train_s": train_s,
        "train_examples_per_s": cfg.steps * cfg.batch_size / train_s,
        "wall_s": time.perf_counter() - start,
        "initial_noise_mse": initial_mse,
        "best_step": best_step,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "threads": cfg.threads,
        **scores,
    }
    write_json(cfg.out_dir / "summary.json", summary)
    print(f"SWD {scores['swd']:.4f} | Gaussian baseline {scores['gaussian_swd']:.4f} -> {cfg.out_dir}")
    return summary


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
