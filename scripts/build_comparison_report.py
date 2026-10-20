"""Archive completed runs and render their measured results, without editing the root README."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro

from xdiffusion.common import write_json


@dataclass
class Config:
    runs: tuple[Path, ...]
    report: Path  # a fresh directory; dated reports are immutable


def main(cfg: Config) -> None:
    if not cfg.runs or len({path.name for path in cfg.runs}) != len(cfg.runs):
        raise ValueError("provide runs with unique directory names")
    if cfg.report.exists():
        raise FileExistsError("choose a fresh report directory; existing evidence is never overwritten")
    rows = []
    for path in cfg.runs:
        for name in ("config.json", "summary.json", "history.json", "samples.png"):
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
        summary = json.loads((path / "summary.json").read_text())
        config = json.loads((path / "config.json").read_text())
        required = (
            "task",
            "family",
            "seed",
            "steps",
            "batch_size",
            "samples",
            "generator_params",
            "nfe",
            "swd",
        )
        if any(key not in summary for key in required):
            raise ValueError(f"{path}: expected a completed family-run summary")
        if summary.get("checkpoint_selection") != "final_ema_at_fixed_budget":
            raise ValueError(f"{path}: this exporter requires final-EMA runs")
        if summary["steps"] != config["steps"]:
            raise ValueError(f"{path}: summary and configured training budget differ")
        rows.append({"run": path.name, **summary})
    cfg.report.mkdir(parents=True)
    for path in cfg.runs:
        destination = cfg.report / "runs" / path.name
        destination.mkdir(parents=True)
        for name in ("config.json", "summary.json", "history.json", "samples.png", "source-sha256.json"):
            if (path / name).exists():
                shutil.copy2(path / name, destination / name)
        # Source provenance comes from training time, never the report builder's current checkout.
    write_json(cfg.report / "results.json", rows)
    text = [
        "# Measured generative models",
        "",
        "Each row is a completed run using final EMA weights. Budgets and model sizes are shown",
        "explicitly; equal updates do not imply equal compute. Inspect configurations before",
        "comparing rows. Image feature distance is dataset-specific, not Inception FID;",
        "confidence is the fraction with maximum class probability ≥95%, not image accuracy.",
        "",
        "| Run | Task | Family | Seed | Total updates | Batch | Samples | Generator params | "
        "NFE | SWD ↓ | Feature distance ↓ | Confident |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fd = f"{row['feature_fd']:.2f}" if "feature_fd" in row else "—"
        confidence = f"{row['confident_fraction']:.1%}" if "confident_fraction" in row else "—"
        text.append(
            f"| [{row['run']}](runs/{row['run']}/config.json) | {row['task']} | {row['family']} | "
            f"{row['seed']} | {row.get('total_steps', row['steps'])} | {row['batch_size']} | "
            f"{row['samples']} | {row['generator_params']:,} | {row['nfe']} | {row['swd']:.4f} | "
            f"{fd} | {confidence} |"
        )
    text += [
        "",
        "## Unfiltered samples",
        "",
        "Grids contain the first generated outputs. Point plots also show held-out references.",
        "",
    ]
    for row in rows:
        text += [f"### {row['run']}", "", f"![{row['run']}](runs/{row['run']}/samples.png)", ""]
    text += [
        "## Reproducibility",
        "",
        "Per-run folders include the exact config, history, and summary. New runs also include",
        "source hashes recorded at training start. Missing hashes in older runs mean that",
        "their training source is not established by this exporter. Checkpoints and full sample",
        "tensors remain in the original run directories; this report archives plots and metadata.",
        "",
    ]
    (cfg.report / "README.md").write_text("\n".join(text))
    print(f"Archived {len(rows)} runs in {cfg.report}")


if __name__ == "__main__":
    main(tyro.cli(Config))
