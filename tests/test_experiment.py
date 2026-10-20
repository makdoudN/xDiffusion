import hashlib
import json
import shutil
from dataclasses import replace

import pytest
import torch

from xdiffusion.data import DataConfig
from xdiffusion.models import ModelConfig
from xdiffusion.run import TrainConfig as ExperimentConfig
from xdiffusion.run import build_model, generate
from xdiffusion.run import load_run as load_experiment
from xdiffusion.train import main


def tiny_config(path, family, task="ring8"):
    return ExperimentConfig(
        family=family,
        out_dir=path,
        encoder_dir=None,
        steps=4,
        batch_size=4,
        data=DataConfig(task=task, train_size=32, val_size=16),
        model=ModelConfig(width=8, depth=1),
        latent_dim=4,
        samples=4,
        sampling_steps=4,
        sample_batch_size=4,
        log_interval=4,
        checkpoint_interval=2,
        r1_interval=2,
        device="cpu",
        threads=2,
    )


@pytest.mark.parametrize("family", ["diffusion", "flow", "priming", "vae", "gan"])
@pytest.mark.parametrize("task", ["ring8", "shapes"])
def test_each_family_trains_saves_and_samples_its_own_checkpoint(tmp_path, family, task):
    cfg = tiny_config(tmp_path, family, task)
    summary = main(cfg)
    model, saved, shape = load_experiment(tmp_path / "ckpt.pt", torch.device("cpu"))
    generated = generate(model, saved, shape)
    torch.testing.assert_close(generated, torch.load(tmp_path / "samples.pt", weights_only=True))
    assert torch.isfinite(generated).all()
    assert summary["family"] == family and summary["steps"] == 4
    assert summary["train_s"] > 0
    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=True)
    assert checkpoint["schema_version"] == 2
    assert checkpoint["optimizer_recipe"]["generator"]["weight_decay"] == 1e-4
    hashes = json.loads((tmp_path / "source-sha256.json").read_text())
    assert hashes == checkpoint["source_sha256"]
    from pathlib import Path

    assert hashes["train.py"] == hashlib.sha256(Path("src/xdiffusion/train.py").read_bytes()).hexdigest()
    history = json.loads((tmp_path / "history.json").read_text())[-1]
    assert history["grad_norm"] >= 0 and history["lr"] > 0
    if family == "priming":
        assert (tmp_path / "primed.pt").exists()
        assert "mmd_surrogate" in history and history["stage"] == "post"
    elif family == "vae":
        assert history["objective"] == pytest.approx(history["reconstruction_nll"] + cfg.beta * history["kl"])
    elif family == "gan":
        assert history["r1_applied"] and history["r1_penalty"] > 0
        assert history["discriminator_lr"] == cfg.lr
        assert checkpoint["optimizer_recipe"]["discriminator"]["weight_decay"] == 0.01


def test_flow_backbone_does_not_apply_the_epsilon_identity_boundary(tmp_path):
    cfg = tiny_config(tmp_path, "flow")
    model = build_model(cfg, (2,))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    assert torch.equal(model(torch.ones(4, 2), torch.ones(4)), torch.zeros(4, 2))


@pytest.mark.parametrize("family", ["flow", "priming", "gan"])
def test_resume_restores_optimizer_ema_rng_and_stage_transition(tmp_path, monkeypatch, family):
    cfg = tiny_config(tmp_path / "run", family)
    save = torch.save
    checkpoint = tmp_path / "step2.pt"

    def save_with_snapshot(value, path, *args, **kwargs):
        if str(path).endswith("last.pt") and value["step"] == 2:
            save(value, checkpoint)
        return save(value, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", save_with_snapshot)
    main(cfg)
    original = torch.load(cfg.out_dir / "samples.pt", weights_only=True)
    shutil.copyfile(checkpoint, cfg.out_dir / "last.pt")
    main(replace(cfg, resume=True))
    torch.testing.assert_close(
        torch.load(cfg.out_dir / "samples.pt", weights_only=True), original, atol=0, rtol=0
    )


def test_standalone_image_experiment_checks_encoders_before_loading_or_training(tmp_path, monkeypatch):
    from xdiffusion import train

    def unexpected_load(*args, **kwargs):
        pytest.fail("expensive dataset loading began before encoder preflight")

    monkeypatch.setattr(train, "load_data", unexpected_load)
    cfg = replace(
        tiny_config(tmp_path / "run", "diffusion"),
        data=DataConfig(task="mnist"),
        encoder_dir=tmp_path / "missing",
    )
    with pytest.raises(FileNotFoundError, match="provide --encoder-dir"):
        main(cfg)


def test_field_warm_start_is_a_new_segment_with_explicit_total_budget(tmp_path):
    first = tiny_config(tmp_path / "first", "flow")
    main(first)
    second = replace(first, out_dir=tmp_path / "second", init_checkpoint=first.out_dir / "ckpt.pt")
    result = main(second)
    assert result["steps"] == 4 and result["parent_steps"] == 4 and result["total_steps"] == 8
    state = torch.load(second.out_dir / "ckpt.pt", weights_only=True)
    assert state["total_steps"] == 8
    (first.out_dir / "ckpt.pt").rename(first.out_dir / "archived.pt")
    resumed = main(replace(second, resume=True))
    assert resumed["total_steps"] == 8


def test_encoder_preflight_validates_metadata_without_advancing_model_rng(tmp_path):
    from xdiffusion.quality import FeatureNet
    from xdiffusion.run import check_encoders

    torch.save(
        {"model": FeatureNet().state_dict(), "task": "mnist", "data_seed": 1234}, tmp_path / "judge.pt"
    )
    cfg = replace(
        tiny_config(tmp_path / "run", "diffusion"), data=DataConfig(task="mnist"), encoder_dir=tmp_path
    )
    state = torch.get_rng_state().clone()
    check_encoders(cfg)
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match="encoder is for"):
        check_encoders(replace(cfg, data=DataConfig(task="fashion")))
