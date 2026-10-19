"""Print per-run metrics, followed by mean ± sample standard deviation by task."""

import json
import statistics
from pathlib import Path

import tyro


def print_results(results: list[dict]) -> None:
    print(
        f"{'task':13} {'seed':>4} {'params':>9} {'noise MSE':>10} "
        f"{'SWD':>9} {'ref SWD':>9} {'train/s':>10} {'gen/s':>9}"
    )
    for r in results:
        print(
            f"{r['task']:13} {r['seed']:4} {r['params']:9,} {r['noise_mse']:10.4f} "
            f"{r['swd']:9.4f} {r['reference_swd']:9.4f} {r['train_examples_per_s']:10.0f} "
            f"{r['samples_per_s']:9.1f}"
        )
    for task in sorted({r["task"] for r in results}):
        rows = [r for r in results if r["task"] == task]
        values = [r["swd"] for r in rows]
        if len(values) > 1:
            print(
                f"{task}: SWD {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} "
                f"(n={len(values)})"
            )
    print("Compare quality within a task; compare speed on the same hardware and sampling settings.")


def main(runs: tyro.conf.Positional[tuple[Path, ...]]) -> None:
    print_results([json.loads((run / "summary.json").read_text()) for run in runs])


if __name__ == "__main__":
    tyro.cli(main)
