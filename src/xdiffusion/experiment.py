"""Compatibility names for the former comparison entry point; use train and generate."""

import tyro

from xdiffusion.run import (
    TrainConfig as ExperimentConfig,
)
from xdiffusion.run import (
    build_model,
    config_from_dict,
    generate,
    score_samples,
)
from xdiffusion.run import (
    load_run as load_experiment,
)
from xdiffusion.train import main

__all__ = [
    "ExperimentConfig",
    "build_model",
    "config_from_dict",
    "generate",
    "load_experiment",
    "score_samples",
    "main",
]

if __name__ == "__main__":
    main(tyro.cli(ExperimentConfig))
