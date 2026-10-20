import shutil
from dataclasses import replace

import pytest
import torch

from xdiffusion.data import DataConfig
from xdiffusion.experiment import ExperimentConfig, build_model, generate, load_experiment, main
from xdiffusion.models import ModelConfig


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
    if family == "priming":
        assert (tmp_path / "primed.pt").exists()


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
