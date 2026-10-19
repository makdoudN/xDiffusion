"""Run a curriculum sequentially, with multiple seeds and ordinary TrainConfigs."""

from dataclasses import dataclass
from pathlib import Path

import tyro

from xdiffusion.common import write_json
from xdiffusion.compare import print_results
from xdiffusion.data import IMAGE_TASKS, POINT_TASKS, DataConfig, Task, load_data
from xdiffusion.models import ModelConfig
from xdiffusion.sample import SamplingConfig
from xdiffusion.train import TrainConfig
from xdiffusion.train import main as train

# task: (optimizer steps, batch size, model width, learning rate)
# These are starting budgets, not claims of converged or equal-compute results.
PRESETS: dict[Task, tuple[int, int, int, float]] = {
    "gaussian": (1500, 256, 128, 1e-3),
    "ring8": (3000, 256, 128, 1e-3),
    "moons": (3000, 256, 128, 1e-3),
    "checkerboard": (5000, 256, 128, 1e-3),
    "shapes": (5000, 64, 32, 3e-4),
    "mnist": (20_000, 64, 32, 3e-4),
    "fashion": (30_000, 64, 32, 3e-4),
    "cifar10": (100_000, 64, 64, 2e-4),
}


@dataclass
class BenchmarkConfig:
    tasks: tuple[Task, ...] = (*POINT_TASKS, "shapes")  # default suite needs no downloads
    seeds: tuple[int, ...] = (0, 1, 2)
    out_dir: Path = Path("out/benchmarks")
    data_root: Path = Path("data")
    steps: int | None = None  # override every task's budget for a smoke run
    samples: int = 512
    sampling_steps: int = 50
    device: str = "auto"


def main(cfg: BenchmarkConfig) -> None:
    if not cfg.tasks or not cfg.seeds:
        raise ValueError("provide at least one task and seed")
    if len(set(cfg.tasks)) != len(cfg.tasks) or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("tasks and seeds must not contain duplicates")
    # Fail before spending compute if any requested output would overwrite a run.
    for task in cfg.tasks:
        for seed in cfg.seeds:
            if (cfg.out_dir / f"{task}-seed{seed}" / "config.json").exists():
                raise FileExistsError("benchmark run already exists; choose a new --out-dir")
    # Check downloads up front, rather than discover a missing dataset hours into a suite.
    for task in cfg.tasks:
        if task in IMAGE_TASKS:
            load_data(DataConfig(task=task, root=cfg.data_root), "val")
    results = []
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for task in cfg.tasks:
        steps, batch, width, lr = PRESETS[task]
        for seed in cfg.seeds:
            run_cfg = TrainConfig(
                data=DataConfig(task=task, root=cfg.data_root),
                model=ModelConfig(width=width),
                sampling=SamplingConfig(samples=cfg.samples, steps=cfg.sampling_steps),
                steps=steps if cfg.steps is None else cfg.steps,
                batch_size=batch,
                lr=lr,
                seed=seed,
                device=cfg.device,
                out_dir=cfg.out_dir / f"{task}-seed{seed}",
            )
            results.append(train(run_cfg))
            write_json(cfg.out_dir / "results.json", results)
    print_results(results)


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
