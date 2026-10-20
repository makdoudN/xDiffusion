import json
from dataclasses import replace

import pytest
import torch

from xdiffusion.common import get_device, load_checkpoint
from xdiffusion.data import DataConfig
from xdiffusion.evaluate import EvaluateConfig
from xdiffusion.evaluate import main as evaluate
from xdiffusion.models import ModelConfig
from xdiffusion.sample import SampleConfig, SamplingConfig, draw_samples
from xdiffusion.sample import main as sample
from xdiffusion.train import TrainConfig
from xdiffusion.train import main as train


def tiny_config(out_dir, task="ring8"):
    return TrainConfig(
        out_dir=out_dir,
        data=DataConfig(task=task, train_size=128, val_size=64),
        model=ModelConfig(width=16, depth=1),
        diffusion_steps=8,
        samples=16,
        sample_batch_size=8,
        sampling_steps=4,
        steps=4,
        batch_size=8,
        checkpoint_interval=2,
        log_interval=4,
        device="cpu",
        threads=2,
    )


@pytest.mark.parametrize("task", ["ring8", "shapes"])
def test_train_checkpoint_sample_and_test_evaluation_round_trip(tmp_path, task):
    cfg = tiny_config(tmp_path / "train", task)
    summary = train(cfg)
    assert summary["split"] == "val" and summary["steps"] * summary["batch_size"] == 32
    assert summary["train_s"] > 0 and summary["samples_per_s"] > 0
    for name in ("config.json", "summary.json", "history.json", "ckpt.pt", "samples.pt", "samples.png"):
        assert (cfg.out_dir / name).stat().st_size > 0
    model, diffusion, checkpoint = load_checkpoint(cfg.out_dir / "ckpt.pt", get_device("cpu"))
    generated = draw_samples(
        model, diffusion, tuple(checkpoint["shape"]), SamplingConfig(samples=16, batch_size=8, steps=4)
    )
    torch.testing.assert_close(generated, torch.load(cfg.out_dir / "samples.pt", weights_only=True))
    evaluate(
        EvaluateConfig(
            checkpoint=cfg.out_dir / "ckpt.pt",
            out_dir=tmp_path / "test",
            sampling=SamplingConfig(samples=16, batch_size=8, steps=4),
            device="cpu",
            threads=2,
        )
    )
    assert json.loads((tmp_path / "test/evaluation.json").read_text())["split"] == "test"
    sample(
        SampleConfig(
            checkpoint=cfg.out_dir / "ckpt.pt",
            out_dir=tmp_path / "sample",
            sampling=SamplingConfig(samples=16, batch_size=8, steps=4),
            device="cpu",
            threads=2,
        )
    )
    assert (tmp_path / "sample/samples.png").exists()
    with pytest.raises(FileExistsError):
        train(cfg)


def test_checkpoint_and_logging_frequency_do_not_change_training(tmp_path):
    cfg = tiny_config(tmp_path / "a")
    train(cfg)
    train(replace(cfg, out_dir=tmp_path / "b", checkpoint_interval=1, log_interval=1))
    a = torch.load(cfg.out_dir / "samples.pt", weights_only=True)
    b = torch.load(tmp_path / "b/samples.pt", weights_only=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_generate_accepts_model_only_legacy_checkpoints_without_loading_data(tmp_path, monkeypatch):
    from dataclasses import asdict

    from xdiffusion import generate as command
    from xdiffusion.models import make_model
    from xdiffusion.run import load_run

    model_cfg = ModelConfig(width=8, depth=1)
    model = make_model((2,), model_cfg)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model_cfg),
            "diffusion_config": {"steps": 4},
            "shape": (2,),
        },
        checkpoint,
    )

    def no_data(*args, **kwargs):
        pytest.fail("sampling-only generation loaded a dataset")

    monkeypatch.setattr(command, "load_data", no_data)
    command.main(
        command.GenerateConfig(
            checkpoint=checkpoint,
            out_dir=tmp_path / "generated",
            samples=4,
            batch_size=2,
            evaluate=False,
            device="cpu",
        )
    )
    assert (tmp_path / "generated/samples.png").exists()
    with pytest.raises(ValueError, match="lacks data metadata"):
        load_run(checkpoint, torch.device("cpu"), require_data=True)


def test_experiment_name_is_a_compatibility_alias():
    from xdiffusion.experiment import ExperimentConfig, main

    assert ExperimentConfig is TrainConfig and main is train
