"""One explicit, fixed-budget comparison loop for five generative families.

Final EMA checkpoints are evaluated by the same code, with the same splits and
sampling seed. Objective values are logged but never compared across families.
GAN steps include a discriminator update; Priming counts both stages in steps.
"""

import json
import math
import platform
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import torch
import tyro
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.common import get_device, save_plot, synchronize, write_json
from xdiffusion.data import DataConfig, load_data
from xdiffusion.diffusion import Diffusion, DiffusionConfig
from xdiffusion.flow import FlowConfig, flow_loss, integrate
from xdiffusion.gan import Discriminator, discriminator_loss, generator_loss
from xdiffusion.metrics import ring8_scores, sliced_wasserstein
from xdiffusion.models import ModelConfig, make_model
from xdiffusion.priming import FrozenFeatures, Priming, laplacian_mmd, reconstruction_loss
from xdiffusion.quality import image_scores, load_encoder
from xdiffusion.vae import VAE, Decoder, vae_loss

Family = Literal["diffusion", "flow", "priming", "vae", "gan"]


@dataclass
class ExperimentConfig:
    family: Family = "flow"
    data: DataConfig = field(default_factory=lambda: DataConfig(task="mnist"))
    model: ModelConfig = field(default_factory=lambda: ModelConfig(width=32))
    flow: FlowConfig = field(default_factory=FlowConfig)
    out_dir: Path = Path("out/flow-mnist")
    encoder_dir: Path | None = Path("out/encoders/mnist")  # judge.pt for evaluation, priming.pt for training
    steps: int = 20_000  # total optimizer steps, including both Priming stages
    batch_size: int = 64
    lr: float = 3e-4
    latent_dim: int = 32  # VAE and GAN
    beta: float = 1.0  # VAE KL coefficient, with reconstruction summed over pixels
    priming_fraction: float = 0.5
    priming_level: float = 0.8
    post_lr: float = 1e-4  # fresh optimizer at the Priming stage transition
    r1_gamma: float = 1.0  # GAN real-data gradient penalty
    r1_interval: int = 16
    ema_decay: float = 0.995
    diffusion_steps: int = 200
    prediction: Literal["epsilon", "v"] = "epsilon"
    sampling_steps: int = 50  # DDIM; flow.steps with Heun uses twice that many evaluations
    samples: int = 1024
    sample_batch_size: int = 128
    sample_seed: int = 0
    log_interval: int = 500
    checkpoint_interval: int = 2000
    seed: int = 0
    device: str = "auto"
    threads: int = 4
    resume: bool = False  # resume the same configured budget, not a different LR schedule

    def __post_init__(self):
        if (
            min(
                self.steps,
                self.batch_size,
                self.latent_dim,
                self.samples,
                self.sample_batch_size,
                self.log_interval,
                self.checkpoint_interval,
                self.r1_interval,
                self.threads,
            )
            < 1
        ):
            raise ValueError("steps, sizes, and intervals must be positive")
        if not 0 <= self.ema_decay < 1 or self.lr <= 0 or self.post_lr <= 0 or self.beta < 0:
            raise ValueError("invalid optimizer settings")
        if not 0 < self.priming_fraction < 1 or not 0 < self.priming_level < 1:
            raise ValueError("Priming fraction and noise level must be in (0, 1)")
        if self.family == "priming" and self.steps < 2:
            raise ValueError("Priming needs at least one step in each stage")
        if not 1 <= self.sampling_steps <= self.diffusion_steps or self.samples < 2:
            raise ValueError("invalid sampling settings")


def build_model(cfg: ExperimentConfig, shape: tuple[int, ...]) -> nn.Module:
    if cfg.family in ("diffusion", "flow"):
        model_cfg = (
            replace(cfg.model, identity_boundary=False)
            if cfg.family == "flow" or cfg.prediction == "v"
            else cfg.model
        )
        return make_model(shape, model_cfg)
    if cfg.family == "priming":
        return Priming(shape, cfg.model)
    if cfg.family == "vae":
        return VAE(shape, cfg.model.width, cfg.latent_dim)
    if cfg.family == "gan":
        return Decoder(shape, cfg.model.width, cfg.latent_dim)
    raise ValueError(f"unknown family: {cfg.family}")


def config_from_dict(values: dict) -> ExperimentConfig:
    return ExperimentConfig(
        **{
            **values,
            "data": DataConfig(**{**values["data"], "root": Path(values["data"]["root"])}),
            "model": ModelConfig(**values["model"]),
            "flow": FlowConfig(**values["flow"]),
            "out_dir": Path(values["out_dir"]),
            "encoder_dir": Path(values["encoder_dir"]) if values["encoder_dir"] is not None else None,
        }
    )


def load_experiment(path: Path, device: torch.device) -> tuple[nn.Module, ExperimentConfig, tuple[int, ...]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = config_from_dict(checkpoint["config"])
    shape = tuple(checkpoint["shape"])
    model = build_model(cfg, shape).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), cfg, shape


@torch.no_grad()
def generate(
    model: nn.Module, cfg: ExperimentConfig, shape: tuple[int, ...], seed: int | None = None
) -> Float[Tensor, " sample *shape"]:
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(cfg.sample_seed if seed is None else seed)
    model.eval()
    diffusion = Diffusion(DiffusionConfig(cfg.diffusion_steps, cfg.prediction)).to(device)
    batches = []
    for start in range(0, cfg.samples, cfg.sample_batch_size):
        n = min(cfg.sample_batch_size, cfg.samples - start)
        if cfg.family == "diffusion":
            output = diffusion.sample(model, (n, *shape), rng, steps=cfg.sampling_steps, clip=len(shape) == 3)
        elif cfg.family == "flow":
            output = integrate(model, torch.randn(n, *shape, device=device, generator=rng), cfg.flow)
        elif cfg.family == "priming":
            output = model(torch.randn(n, *shape, device=device, generator=rng))
        else:
            decoder = model.decoder if isinstance(model, VAE) else model
            output = decoder(torch.randn(n, cfg.latent_dim, device=device, generator=rng))
        if len(shape) == 3:
            output = output.clamp(-1, 1)
        batches.append(output.cpu())
    return torch.cat(batches)


@torch.no_grad()
def score_samples(
    samples: Float[Tensor, " sample *shape"],
    real: Float[Tensor, " dataset *shape"],
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict:
    n = len(samples)
    if len(real) < 2 * n:
        raise ValueError("need two disjoint reference sets, each as large as the generated sample set")
    scores = {
        "swd": sliced_wasserstein(real[:n], samples),
        "reference_swd": sliced_wasserstein(real[:n], real[n : 2 * n]),
    }
    if cfg.data.task == "ring8":
        scores.update(ring8_scores(samples))
    if cfg.encoder_dir is not None and cfg.data.task in ("mnist", "fashion"):
        judge = load_encoder(cfg.encoder_dir / "judge.pt", device, cfg.data.task, cfg.data.seed)
        scores.update(image_scores(judge, real[:n], samples))
        scores["reference_feature_fd"] = image_scores(judge, real[:n], real[n : 2 * n])["feature_fd"]
    return scores


def update(loss: Float[Tensor, ""], optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def main(cfg: ExperimentConfig) -> dict:
    torch.set_num_threads(cfg.threads)
    if (cfg.out_dir / "config.json").exists() and not cfg.resume:
        raise FileExistsError(f"run already exists: {cfg.out_dir}; choose a new --out-dir")
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    if cfg.encoder_dir is not None and cfg.data.task in ("mnist", "fashion"):
        required = ["judge.pt", "priming.pt"] if cfg.family == "priming" else ["judge.pt"]
        for filename in required:
            if not (cfg.encoder_dir / filename).is_file():
                raise FileNotFoundError(
                    f"missing {cfg.encoder_dir / filename}; train encoders with xdiffusion.quality "
                    "or provide --encoder-dir before starting this experiment"
                )
    train, val = load_data(cfg.data), load_data(cfg.data, "val")
    if 2 * cfg.samples > len(val):
        raise ValueError("validation data must contain at least 2 * samples")
    shape = tuple(train.shape[1:])
    model = build_model(cfg, shape).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(DiffusionConfig(cfg.diffusion_steps, cfg.prediction)).to(device)
    betas = (0.5, 0.999) if cfg.family == "gan" else (0.9, 0.999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=betas, weight_decay=1e-4)
    critic = Discriminator(shape, cfg.model.width).to(device) if cfg.family == "gan" else None
    critic_optimizer = (
        torch.optim.AdamW(critic.parameters(), lr=cfg.lr, betas=betas) if critic is not None else None
    )
    features = None
    feature_scale = 1.0
    if cfg.family == "priming" and cfg.data.task in ("mnist", "fashion"):
        if cfg.encoder_dir is None:
            raise ValueError("Priming on real images needs its frozen training encoder")
        encoder = load_encoder(cfg.encoder_dir / "priming.pt", device, cfg.data.task, cfg.data.seed)
        features = FrozenFeatures(encoder, train[:512].to(device))
        feature_scale = 1 / math.sqrt(features(train[:2].to(device)).shape[-1])

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(asdict(cfg), default=str))
    data_rng = torch.Generator().manual_seed(cfg.seed)
    noise_rng = torch.Generator(device=device).manual_seed(cfg.seed)
    boundary = max(1, min(cfg.steps - 1, int(cfg.steps * cfg.priming_fraction)))
    history, first_step, train_s = [], 1, 0.0

    if cfg.resume:
        state = torch.load(cfg.out_dir / "last.pt", map_location=device, weights_only=True)
        previous = json.loads(json.dumps(asdict(config_from_dict(state["config"])), default=str))
        previous["resume"] = True
        if config != previous:
            raise ValueError("resuming requires the exact original configuration and budget")
        model.load_state_dict(state["online"])
        ema.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if critic is not None:
            critic.load_state_dict(state["critic"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        data_rng.set_state(state["data_rng"].cpu())
        noise_rng.set_state(state["noise_rng"].cpu())
        history, first_step, train_s = state["history"], state["step"] + 1, state["train_s"]

    write_json(cfg.out_dir / "config.json", config)

    def checkpoint(step: int, filename: str) -> None:
        state = {
            "family": cfg.family,
            "config": config,
            "shape": shape,
            "step": step,
            "model": ema.state_dict(),
            "online": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_rng": data_rng.get_state(),
            "noise_rng": noise_rng.get_state(),
            "history": history,
            "train_s": train_s,
        }
        if critic is not None:
            state.update(critic=critic.state_dict(), critic_optimizer=critic_optimizer.state_dict())
        torch.save(state, cfg.out_dir / filename)

    print(
        f"{cfg.family} / {cfg.data.task} | "
        f"{sum(p.numel() for p in model.parameters()):,} parameters | {device}",
        flush=True,
    )
    synchronize(device)
    segment_start = time.perf_counter()
    for step in range(first_step, cfg.steps + 1):
        model.train()
        if cfg.family == "priming" and step == boundary + 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.post_lr, weight_decay=1e-4)
        base_lr = cfg.post_lr if cfg.family == "priming" and step > boundary else cfg.lr
        # Stage-local warmup and cosine decay; the configured budget is unchanged on resume.
        stage_start = boundary if cfg.family == "priming" and step > boundary else 0
        stage_end = boundary if cfg.family == "priming" and step <= boundary else cfg.steps
        progress = (step - stage_start) / max(1, stage_end - stage_start)
        rate = base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate * min(1.0, (step - stage_start) / min(100, stage_end - stage_start))
        indices = torch.randint(len(train), (cfg.batch_size,), generator=data_rng)
        clean = train[indices].to(device)
        noise = torch.randn(clean.shape, generator=noise_rng, device=device)
        if cfg.family == "diffusion":
            times = torch.randint(
                1, cfg.diffusion_steps + 1, (len(clean),), generator=noise_rng, device=device
            )
            loss = diffusion.loss(model, clean, times, noise)
        elif cfg.family == "flow":
            times = torch.rand(len(clean), generator=noise_rng, device=device)
            loss = flow_loss(model, clean, noise, times)
        elif cfg.family == "priming":
            if step <= boundary:
                loss = reconstruction_loss(model, clean, noise, cfg.priming_level)
            else:
                generated = model(noise)
                with torch.no_grad():
                    negative = model(torch.randn(clean.shape, generator=noise_rng, device=device))
                encode = features if features is not None else nn.Flatten(start_dim=1)
                loss = laplacian_mmd(encode(generated), encode(clean), encode(negative), feature_scale)
        elif cfg.family == "vae":
            latent = torch.randn(len(clean), cfg.latent_dim, generator=noise_rng, device=device)
            loss = vae_loss(model, clean, latent, cfg.beta)
        else:
            z = torch.randn(len(clean), cfg.latent_dim, generator=noise_rng, device=device)
            fake = model(z)
            critic.requires_grad_(True)
            clean.requires_grad_(step % cfg.r1_interval == 0 and cfg.r1_gamma > 0)
            real_logits = critic(clean)
            d_loss = discriminator_loss(real_logits, critic(fake.detach()))
            if clean.requires_grad:
                gradient = torch.autograd.grad(real_logits.sum(), clean, create_graph=True)[0]
                d_loss = (
                    d_loss + cfg.r1_gamma / 2 * cfg.r1_interval * gradient.flatten(1).square().sum(-1).mean()
                )
            update(d_loss, critic_optimizer, critic)
            critic.requires_grad_(False)
            loss = generator_loss(critic(fake))
        update(loss, optimizer, model)
        with torch.no_grad():
            decay = min(cfg.ema_decay, (1 + step) / (10 + step))
            for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                average.lerp_(parameter, 1 - decay)
        if step % cfg.log_interval == 0 or step == cfg.steps:
            print(f"step {step:6} | objective {loss.item():.5f}", flush=True)
            history.append(
                {
                    "step": step,
                    "objective": loss.item(),
                    "stage": "post" if cfg.family == "priming" and step > boundary else "train",
                }
            )
        if (
            step % cfg.checkpoint_interval == 0
            or step == cfg.steps
            or (cfg.family == "priming" and step == boundary)
        ):
            synchronize(device)
            train_s += time.perf_counter() - segment_start
            checkpoint(step, "last.pt")
            if cfg.family == "priming" and step == boundary:
                checkpoint(step, "primed.pt")
            write_json(cfg.out_dir / "history.json", history)
            synchronize(device)
            segment_start = time.perf_counter()

    checkpoint(cfg.steps, "ckpt.pt")
    # Warm up the same generation path, then include host transfers in synchronized timing.
    generate(ema, replace(cfg, samples=2), shape)
    synchronize(device)
    start = time.perf_counter()
    samples = generate(ema, cfg, shape)
    synchronize(device)
    sample_s = time.perf_counter() - start
    scores = score_samples(samples, val, cfg, device)
    torch.save(samples, cfg.out_dir / "samples.pt")
    save_plot(samples, cfg.out_dir / "samples.png", val[: cfg.samples])
    summary = {
        "family": cfg.family,
        "task": cfg.data.task,
        "seed": cfg.seed,
        "data_seed": cfg.data.seed,
        "split": "val",
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "samples": cfg.samples,
        "params": sum(p.numel() for p in model.parameters()),
        "generator_params": sum(
            p.numel() for p in (model.decoder if isinstance(model, VAE) else model).parameters()
        ),
        "frozen_training_params": sum(p.numel() for p in features.parameters())
        if features is not None
        else 0,
        "auxiliary_params": sum(p.numel() for p in critic.parameters()) if critic is not None else 0,
        "train_s": train_s,
        "train_examples_per_s": cfg.steps * cfg.batch_size / train_s,
        "sample_s": sample_s,
        "samples_per_s": cfg.samples / sample_s,
        "nfe": cfg.sampling_steps
        if cfg.family == "diffusion"
        else cfg.flow.steps * (2 if cfg.flow.solver == "heun" else 1)
        if cfg.family == "flow"
        else 1,
        "checkpoint_selection": "final_ema_at_fixed_budget",
        "prediction": cfg.prediction if cfg.family == "diffusion" else None,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        **scores,
    }
    write_json(cfg.out_dir / "summary.json", summary)
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    main(tyro.cli(ExperimentConfig))
