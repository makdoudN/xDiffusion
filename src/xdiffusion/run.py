"""Run configuration, model loading, generation, and shared measurement.

Both checkpoint generations enter through load_run; all families use generate.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.data import DataConfig
from xdiffusion.diffusion import Diffusion, DiffusionConfig
from xdiffusion.flow import FlowConfig, integrate
from xdiffusion.metrics import ring8_scores, sliced_wasserstein
from xdiffusion.models import ModelConfig, make_model
from xdiffusion.priming import Priming
from xdiffusion.quality import image_scores, load_encoder
from xdiffusion.vae import VAE, Decoder

Family = Literal["diffusion", "flow", "priming", "vae", "gan"]


@dataclass
class TrainConfig:
    family: Family = "diffusion"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    out_dir: Path = Path("out/ring8")
    encoder_dir: Path | None = None  # judge.pt for evaluation, priming.pt for training
    steps: int = 3000  # total optimizer steps, including both Priming stages
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] | None = None  # resolved by family; effective recipe is saved
    discriminator_lr: float | None = None  # inherit lr when omitted
    discriminator_weight_decay: float = 0.01  # historical recipe, now explicit
    discriminator_betas: tuple[float, float] = (0.5, 0.999)
    discriminator_schedule: Literal["constant", "cosine"] = "constant"
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
    sampler: Literal["ddim", "ddpm"] = "ddim"
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
    init_checkpoint: Path | None = None  # warm-start a field model from EMA, with a fresh optimizer

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
        if self.weight_decay < 0 or self.discriminator_weight_decay < 0 or self.r1_gamma < 0:
            raise ValueError("regularization strengths must be nonnegative")
        if self.discriminator_lr is not None and self.discriminator_lr <= 0:
            raise ValueError("discriminator learning rate must be positive")
        for betas in (self.betas, self.discriminator_betas):
            if betas is not None and (len(betas) != 2 or not all(0 <= beta < 1 for beta in betas)):
                raise ValueError("AdamW requires two betas in [0, 1)")
        if not 0 < self.priming_fraction < 1 or not 0 < self.priming_level < 1:
            raise ValueError("Priming fraction and noise level must be in (0, 1)")
        if self.family == "priming" and self.steps < 2:
            raise ValueError("Priming needs at least one step in each stage")
        if not 1 <= self.sampling_steps <= self.diffusion_steps or self.samples < 2:
            raise ValueError("invalid sampling settings")
        if (
            self.family == "diffusion"
            and self.sampler == "ddpm"
            and self.sampling_steps != self.diffusion_steps
        ):
            raise ValueError("DDPM requires all diffusion steps")
        if self.init_checkpoint is not None and self.family not in ("diffusion", "flow"):
            raise ValueError("warm starts currently support diffusion and flow field models")


def check_encoders(cfg: TrainConfig) -> None:
    """Reject missing or incompatible evaluation/training assets before spending compute."""
    if cfg.data.task not in ("mnist", "fashion"):
        return
    if cfg.encoder_dir is None:
        if cfg.family == "priming":
            raise ValueError("Priming on real images needs its frozen training encoder")
        return
    required = ("judge.pt", "priming.pt") if cfg.family == "priming" else ("judge.pt",)
    for filename in required:
        path = cfg.encoder_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; train encoders or provide --encoder-dir")
        # Loading a module initializes temporary weights: preflight must preserve the model RNG.
        with torch.random.fork_rng(devices=[]):
            load_encoder(path, torch.device("cpu"), cfg.data.task, cfg.data.seed)


def optimizer_recipes(cfg: TrainConfig) -> dict:
    """Effective AdamW arguments, including the historically different GAN critic."""
    generator = {
        "lr": cfg.lr,
        "betas": cfg.betas or ((0.5, 0.999) if cfg.family == "gan" else (0.9, 0.999)),
        "weight_decay": cfg.weight_decay,
    }
    recipes = {"generator": generator}
    if cfg.family == "priming":
        recipes["post"] = {**generator, "lr": cfg.post_lr}
    if cfg.family == "gan":
        recipes["discriminator"] = {
            "lr": cfg.discriminator_lr if cfg.discriminator_lr is not None else cfg.lr,
            "betas": cfg.discriminator_betas,
            "weight_decay": cfg.discriminator_weight_decay,
        }
    return recipes


def build_model(cfg: TrainConfig, shape: tuple[int, ...]) -> nn.Module:
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


def config_from_dict(values: dict) -> TrainConfig:
    return TrainConfig(
        **{
            **values,
            "data": DataConfig(**{**values["data"], "root": Path(values["data"]["root"])}),
            "model": ModelConfig(**values["model"]),
            "flow": FlowConfig(**values["flow"]),
            "betas": tuple(values["betas"]) if values.get("betas") else None,
            "discriminator_betas": tuple(values.get("discriminator_betas", (0.5, 0.999))),
            "out_dir": Path(values["out_dir"]),
            "encoder_dir": Path(values["encoder_dir"]) if values["encoder_dir"] is not None else None,
            "init_checkpoint": Path(values["init_checkpoint"]) if values.get("init_checkpoint") else None,
        }
    )


def load_run(
    path: Path, device: torch.device, require_data: bool = False
) -> tuple[nn.Module, TrainConfig, tuple[int, ...]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "family" in checkpoint:
        cfg = config_from_dict(checkpoint["config"])
    else:
        # Original checkpoints may contain only model metadata: those support sampling, not scoring.
        if require_data and "data_config" not in checkpoint:
            raise ValueError("legacy checkpoint lacks data metadata; use --no-evaluate for sampling only")
        data = checkpoint.get("data_config", {})
        cfg = TrainConfig(
            family="diffusion",
            model=ModelConfig(**{"architecture": "tiny", **checkpoint["model_config"]}),
            data=DataConfig(**{**data, "root": Path(data.get("root", "data"))}),
            diffusion_steps=checkpoint["diffusion_config"]["steps"],
            prediction=checkpoint["diffusion_config"].get("prediction", "epsilon"),
            sampling_steps=min(50, checkpoint["diffusion_config"]["steps"]),
        )
    shape = tuple(checkpoint["shape"])
    model = build_model(cfg, shape).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), cfg, shape


@torch.no_grad()
def generate(
    model: nn.Module, cfg: TrainConfig, shape: tuple[int, ...], seed: int | None = None
) -> Float[Tensor, " sample *shape"]:
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(cfg.sample_seed if seed is None else seed)
    model.eval()
    diffusion = Diffusion(DiffusionConfig(cfg.diffusion_steps, cfg.prediction)).to(device)
    batches = []
    for start in range(0, cfg.samples, cfg.sample_batch_size):
        n = min(cfg.sample_batch_size, cfg.samples - start)
        if cfg.family == "diffusion":
            output = diffusion.sample(
                model, (n, *shape), rng, sampler=cfg.sampler, steps=cfg.sampling_steps, clip=len(shape) == 3
            )
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
    cfg: TrainConfig,
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
