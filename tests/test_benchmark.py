import pytest

from xdiffusion import benchmark


def test_missing_image_data_is_detected_before_any_training(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise RuntimeError("dataset not downloaded")

    def should_not_train(*args, **kwargs):
        pytest.fail("training started before all datasets were checked")

    monkeypatch.setattr(benchmark, "load_data", missing)
    monkeypatch.setattr(benchmark, "train", should_not_train)
    cfg = benchmark.BenchmarkConfig(tasks=("gaussian", "mnist"), seeds=(0,), out_dir=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="dataset not downloaded"):
        benchmark.main(cfg)
    assert not cfg.out_dir.exists()
