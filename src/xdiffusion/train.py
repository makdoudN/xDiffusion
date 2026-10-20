"""The single training loop: explicit objectives, final EMA, and exact-budget resume."""

import hashlib
import json
import math
import platform
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch
import tyro
from jaxtyping import Float
from torch import Tensor, nn

from xdiffusion.common import get_device, save_plot, synchronize, write_json
from xdiffusion.data import load_data
from xdiffusion.diffusion import Diffusion, DiffusionConfig
from xdiffusion.flow import flow_loss
from xdiffusion.gan import Discriminator, discriminator_loss, generator_loss
from xdiffusion.priming import FrozenFeatures, laplacian_mmd, reconstruction_loss
from xdiffusion.quality import load_encoder
from xdiffusion.run import (
    TrainConfig,
    build_model,
    check_encoders,
    config_from_dict,
    generate,
    optimizer_recipes,
    score_samples,
)
from xdiffusion.vae import VAE, vae_terms


def update(loss: Float[Tensor, ""], optimizer: torch.optim.Optimizer, model: nn.Module) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return norm


def main(cfg: TrainConfig) -> dict:
    torch.set_num_threads(cfg.threads)
    if (cfg.out_dir / "config.json").exists() and not cfg.resume:
        raise FileExistsError(f"run already exists: {cfg.out_dir}; choose a new --out-dir")
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    check_encoders(cfg)
    train, val = load_data(cfg.data), load_data(cfg.data, "val")
    if 2 * cfg.samples > len(val):
        raise ValueError("validation data must contain at least 2 * samples")
    shape = tuple(train.shape[1:])
    model = build_model(cfg, shape).to(device)
    parent_steps, parent_train_s = 0, 0.0
    if cfg.init_checkpoint is not None and not cfg.resume:
        parent = torch.load(cfg.init_checkpoint, map_location=device, weights_only=True)
        if parent["family"] != cfg.family or tuple(parent["shape"]) != shape:
            raise ValueError("warm-start family and data shape must match")
        if cfg.family == "diffusion" and parent["config"].get("prediction", "epsilon") != cfg.prediction:
            raise ValueError("warm-start prediction target must match")
        model.load_state_dict(parent["model"])
        parent_steps = parent.get("total_steps", parent["step"])
        parent_train_s = parent.get("total_train_s", parent["train_s"])
        del parent
    ema = deepcopy(model).eval().requires_grad_(False)
    diffusion = Diffusion(DiffusionConfig(cfg.diffusion_steps, cfg.prediction)).to(device)
    recipes = optimizer_recipes(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), **recipes["generator"])
    critic = Discriminator(shape, cfg.model.width).to(device) if cfg.family == "gan" else None
    critic_optimizer = (
        torch.optim.AdamW(critic.parameters(), **recipes["discriminator"]) if critic is not None else None
    )
    features = None
    feature_scale = 1.0
    if cfg.family == "priming" and cfg.data.task in ("mnist", "fashion"):
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
        parent_steps = state.get("total_steps", state["step"]) - state["step"]
        parent_train_s = state.get("total_train_s", state["train_s"]) - state["train_s"]

    write_json(cfg.out_dir / "config.json", config)
    sources = list(sorted(Path(__file__).parent.glob("*.py")))
    root = Path(__file__).resolve().parents[2]
    sources += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file()]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(cfg.out_dir / "source-sha256.json", source_hashes)

    def checkpoint(step: int, filename: str) -> None:
        state = {
            "schema_version": 2,
            "source_sha256": source_hashes,
            "optimizer_recipe": recipes,
            "family": cfg.family,
            "config": config,
            "shape": shape,
            "step": step,
            "total_steps": parent_steps + step,
            "total_train_s": parent_train_s + train_s,
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
            optimizer = torch.optim.AdamW(model.parameters(), **recipes["post"])
        base_lr = cfg.post_lr if cfg.family == "priming" and step > boundary else cfg.lr
        # Stage-local warmup and cosine decay; the configured budget is unchanged on resume.
        stage_start = boundary if cfg.family == "priming" and step > boundary else 0
        stage_end = boundary if cfg.family == "priming" and step <= boundary else cfg.steps
        progress = (step - stage_start) / max(1, stage_end - stage_start)
        rate = base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate * min(1.0, (step - stage_start) / min(100, stage_end - stage_start))
        if critic_optimizer is not None:
            d_rate = recipes["discriminator"]["lr"]
            if cfg.discriminator_schedule == "cosine":
                d_rate *= (rate / base_lr) * min(1.0, step / min(100, cfg.steps))
            for group in critic_optimizer.param_groups:
                group["lr"] = d_rate
        diagnostics = {}
        indices = torch.randint(len(train), (cfg.batch_size,), generator=data_rng)
        clean = train[indices].to(device)
        noise = torch.randn(clean.shape, generator=noise_rng, device=device)
        if cfg.family == "diffusion":
            times = torch.randint(
                1, cfg.diffusion_steps + 1, (len(clean),), generator=noise_rng, device=device
            )
            loss = diffusion.loss(model, clean, times, noise)
            diagnostics["prediction_mse"] = loss
        elif cfg.family == "flow":
            times = torch.rand(len(clean), generator=noise_rng, device=device)
            loss = flow_loss(model, clean, noise, times)
            diagnostics["velocity_mse"] = loss
        elif cfg.family == "priming":
            if step <= boundary:
                loss = reconstruction_loss(model, clean, noise, cfg.priming_level)
                diagnostics["reconstruction_mse"] = loss
            else:
                generated = model(noise)
                with torch.no_grad():
                    negative = model(torch.randn(clean.shape, generator=noise_rng, device=device))
                encode = features if features is not None else nn.Flatten(start_dim=1)
                loss = laplacian_mmd(encode(generated), encode(clean), encode(negative), feature_scale)
                diagnostics["mmd_surrogate"] = loss
        elif cfg.family == "vae":
            latent = torch.randn(len(clean), cfg.latent_dim, generator=noise_rng, device=device)
            loss, diagnostics = vae_terms(model, clean, latent, cfg.beta)
        else:
            z = torch.randn(len(clean), cfg.latent_dim, generator=noise_rng, device=device)
            fake = model(z)
            critic.requires_grad_(True)
            clean.requires_grad_(step % cfg.r1_interval == 0 and cfg.r1_gamma > 0)
            real_logits = critic(clean)
            d_loss = discriminator_loss(real_logits, critic(fake.detach()))
            diagnostics["discriminator_logistic"] = d_loss
            diagnostics["r1_penalty"] = clean.new_zeros(())
            if clean.requires_grad:
                gradient = torch.autograd.grad(real_logits.sum(), clean, create_graph=True)[0]
                penalty = cfg.r1_gamma / 2 * cfg.r1_interval * gradient.flatten(1).square().sum(-1).mean()
                diagnostics["r1_penalty"] = penalty
                d_loss = d_loss + penalty
            diagnostics["discriminator_grad_norm"] = update(d_loss, critic_optimizer, critic)
            diagnostics["real_logit"] = real_logits.mean()
            critic.requires_grad_(False)
            fake_logits = critic(fake)
            loss = generator_loss(fake_logits)
            diagnostics["generator_logistic"] = loss
            diagnostics["fake_logit_after_d_update"] = fake_logits.mean()
        diagnostics["grad_norm"] = update(loss, optimizer, model)
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
                    "lr": optimizer.param_groups[0]["lr"],
                    **({"r1_applied": bool(clean.requires_grad)} if critic is not None else {}),
                    **(
                        {"discriminator_lr": critic_optimizer.param_groups[0]["lr"]}
                        if critic is not None
                        else {}
                    ),
                    **{name: value.detach().item() for name, value in diagnostics.items()},
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
        "parent_steps": parent_steps,
        "total_steps": parent_steps + cfg.steps,
        "total_train_s": parent_train_s + train_s,
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
        "optimizer_recipe": recipes,
        "schedules": {"generator": "stage_local_warmup_cosine", "discriminator": cfg.discriminator_schedule}
        if critic is not None
        else {"generator": "stage_local_warmup_cosine"},
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
    main(tyro.cli(TrainConfig))
