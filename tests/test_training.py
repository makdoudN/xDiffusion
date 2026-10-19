import json
from dataclasses import replace

import pytest
import torch

from xdiffusion.common import get_device, load_checkpoint
from xdiffusion.data import DataConfig
from xdiffusion.diffusion import DiffusionConfig
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
        diffusion=DiffusionConfig(steps=8),
        sampling=SamplingConfig(samples=16, batch_size=8, steps=4),
        steps=4,
        batch_size=8,
        eval_interval=2,
        log_interval=4,
        device="cpu",
        threads=2,
    )


@pytest.mark.parametrize("task", ["ring8", "shapes"])
def test_train_checkpoint_sample_and_test_evaluation_round_trip(tmp_path, task):
    cfg = tiny_config(tmp_path / "train", task)
    summary = train(cfg)
    assert summary["split"] == "val" and summary["train_examples"] == 32
    assert summary["train_s"] > 0 and summary["samples_per_s"] > 0
    for name in ("config.json", "summary.json", "history.json", "ckpt.pt", "samples.pt", "samples.png"):
        assert (cfg.out_dir / name).stat().st_size > 0
    model, diffusion, checkpoint = load_checkpoint(cfg.out_dir / "ckpt.pt", get_device("cpu"))
    generated = draw_samples(model, diffusion, tuple(checkpoint["shape"]), cfg.sampling)
    torch.testing.assert_close(generated, torch.load(cfg.out_dir / "samples.pt", weights_only=True))
    evaluate(
        EvaluateConfig(
            checkpoint=cfg.out_dir / "ckpt.pt",
            out_dir=tmp_path / "test",
            sampling=cfg.sampling,
            device="cpu",
            threads=2,
        )
    )
    assert json.loads((tmp_path / "test/evaluation.json").read_text())["split"] == "test"
    sample(
        SampleConfig(
            checkpoint=cfg.out_dir / "ckpt.pt",
            out_dir=tmp_path / "sample",
            sampling=cfg.sampling,
            device="cpu",
            threads=2,
        )
    )
    assert (tmp_path / "sample/samples.png").exists()
    with pytest.raises(FileExistsError):
        train(cfg)


def test_evaluating_more_often_does_not_change_training(tmp_path):
    cfg = tiny_config(tmp_path / "a")
    train(cfg)
    train(replace(cfg, out_dir=tmp_path / "b", eval_interval=1))
    a = json.loads((tmp_path / "a/history.json").read_text())[-1]
    b = json.loads((tmp_path / "b/history.json").read_text())[-1]
    assert a["val_noise_mse"] == b["val_noise_mse"]
