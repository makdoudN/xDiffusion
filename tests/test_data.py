import pytest
import torch

from xdiffusion.data import DataConfig, load_data, synthetic


@pytest.mark.parametrize("task", ["gaussian", "ring8", "moons", "checkerboard", "shapes"])
def test_synthetic_splits_are_repeatable_independent_and_do_not_consume_global_rng(task):
    cfg = DataConfig(task=task, train_size=128, val_size=128)
    state = torch.random.get_rng_state()
    train, val, test = (load_data(cfg, split) for split in ("train", "val", "test"))
    assert torch.equal(train, load_data(cfg))
    assert not torch.equal(train, val)
    assert not torch.equal(val, test)
    assert torch.equal(state, torch.random.get_rng_state())
    expected = (128, 1, 16, 16) if task == "shapes" else (128, 2)
    assert train.shape == expected
    assert torch.isfinite(train).all()


def test_checkerboard_has_only_the_eight_occupied_cells():
    points = synthetic("checkerboard", 10_000, seed=0)
    cells = ((points + 1) * 2).floor().long()
    assert (cells[:, 0] % 2 == cells[:, 1] % 2).all()
    assert len(cells.unique(dim=0)) == 8


def test_ring_has_all_eight_modes_and_small_radial_noise():
    points = synthetic("ring8", 10_000, seed=0)
    assert points.norm(dim=-1).mean().item() == pytest.approx(1.5, abs=0.01)
    angle = torch.atan2(points[:, 1], points[:, 0])
    assert len((angle / (torch.pi / 4)).round().long().remainder(8).unique()) == 8


@pytest.mark.parametrize(
    "task,class_name", [("mnist", "MNIST"), ("fashion", "FashionMNIST"), ("cifar10", "CIFAR10")]
)
def test_official_images_have_disjoint_train_val_and_untouched_test(monkeypatch, task, class_name):
    import torchvision.datasets

    # Stand in for downloaded bytes: exercise actual layout, partition and normalization code.
    shape = (6000, 4, 4, 3) if task == "cifar10" else (6000, 4, 4)
    raw = torch.randint(256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(5))

    class FakeDataset:
        def __init__(self, root, train, download):
            self.data = raw if train else raw[:100]

    monkeypatch.setattr(torchvision.datasets, class_name, FakeDataset)
    cfg = DataConfig(task=task)
    train, val, test = (load_data(cfg, split) for split in ("train", "val", "test"))
    order = torch.randperm(6000, generator=torch.Generator().manual_seed(cfg.seed))
    combined = torch.cat([val, train])
    # Undo the permutation to recover exactly the original training pixels.
    original = combined[order.argsort()]
    expected = raw.permute(0, 3, 1, 2) if task == "cifar10" else raw[:, None]
    torch.testing.assert_close(original, expected.float() / 127.5 - 1)
    torch.testing.assert_close(test, original[:100])
    assert len(train) == 1000 and len(val) == 5000
