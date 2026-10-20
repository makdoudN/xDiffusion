"""Run all five generative families under explicit shared budgets and seeds."""

from dataclasses import dataclass, replace
from pathlib import Path

import tyro

from xdiffusion.common import write_json
from xdiffusion.data import DataConfig, Task, load_data
from xdiffusion.experiment import ExperimentConfig, Family
from xdiffusion.experiment import main as train
from xdiffusion.models import ModelConfig


@dataclass
class StudyConfig:
    families: tuple[Family, ...] = ("diffusion", "flow", "priming", "vae", "gan")
    tasks: tuple[Task, ...] = ("mnist",)
    seeds: tuple[int, ...] = (0,)
    out_dir: Path = Path("out/comparison")
    encoder_root: Path = Path("out/encoders")
    data_root: Path = Path("data")
    steps: int = 20_000
    batch_size: int = 64
    width: int = 32
    samples: int = 1024
    device: str = "auto"


def main(cfg: StudyConfig) -> None:
    if not cfg.families or not cfg.tasks or not cfg.seeds:
        raise ValueError("at least one family, task, and seed is required")
    if any(len(set(values)) != len(values) for values in (cfg.families, cfg.tasks, cfg.seeds)):
        raise ValueError("families, tasks, and seeds must be unique")
    # Check all requested output paths before starting expensive training.
    for task in cfg.tasks:
        load_data(DataConfig(task=task, root=cfg.data_root), "val")
        for family in cfg.families:
            for seed in cfg.seeds:
                if (cfg.out_dir / f"{task}-{family}-seed{seed}" / "config.json").exists():
                    raise FileExistsError("a requested run already exists; choose a fresh --out-dir")
        if task in ("mnist", "fashion"):
            for name in ("judge", "priming"):
                if name == "priming" and "priming" not in cfg.families:
                    continue
                path = cfg.encoder_root / task / f"{name}.pt"
                if not path.exists():
                    raise FileNotFoundError(
                        f"missing {path}; run python -m xdiffusion.quality --data.task {task}"
                    )
    results = []
    for task in cfg.tasks:
        for family in cfg.families:
            for seed in cfg.seeds:
                experiment = ExperimentConfig(
                    family=family,
                    data=DataConfig(task=task, root=cfg.data_root),
                    model=ModelConfig(width=cfg.width),
                    out_dir=cfg.out_dir / f"{task}-{family}-seed{seed}",
                    encoder_dir=cfg.encoder_root / task if task in ("mnist", "fashion") else None,
                    steps=cfg.steps,
                    batch_size=cfg.batch_size,
                    samples=cfg.samples,
                    device=cfg.device,
                    seed=seed,
                )
                if task in ("gaussian", "ring8", "moons", "checkerboard"):
                    experiment = replace(experiment, model=ModelConfig(width=max(64, cfg.width)), lr=1e-3)
                results.append(train(experiment))
                write_json(cfg.out_dir / "results.json", results)
    print(f"{'task':12} {'family':12} {'seed':>4} {'SWD':>8} {'feature FD':>12} {'confident':>10} {'NFE':>4}")
    for r in results:
        fd = f"{r['feature_fd']:.2f}" if "feature_fd" in r else "—"
        confidence = f"{r['confident_fraction']:.1%}" if "confident_fraction" in r else "—"
        print(
            f"{r['task']:12} {r['family']:12} {r['seed']:4} "
            f"{r['swd']:8.4f} {fd:>12} {confidence:>10} {r['nfe']:4}"
        )


if __name__ == "__main__":
    main(tyro.cli(StudyConfig))
