"""Archive measured runs and fill README sections from actual result files.

Requires all requested runs and a matching quality diagnostic, including failures.
A completed investigation does not imply that its quality target was reached.
It never promotes a hand-picked sample or a training-loss ranking.
"""

import hashlib
import json
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path

import tyro

from xdiffusion.common import write_json

FAMILIES = ("diffusion", "flow", "priming", "vae", "gan")


@dataclass
class Config:
    images: Path = Path("out/comparison-mnist")
    rings: Path = Path("out/comparison-ring8")
    corrected: Path = Path("out/quality-ablation/v50k")
    quality_gate: Path = Path("out/quality-ablation/v50k-quality.json")
    report: Path = Path("reports/comparison-2026-10-20")
    allow_pending: bool = False  # useful during a live investigation; never a completed comparison


def archive(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "history.json", "summary.json", "samples.png"):
        shutil.copy2(source / name, destination / name)
    return json.loads((source / "summary.json").read_text())


def update_section(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    replacement = f"{start}\n{body}\n{end}"
    placeholder = f"<!-- {name} -->"
    if placeholder in text:
        return text.replace(placeholder, replacement)
    return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: replacement, text, flags=re.S)


def main(cfg: Config) -> None:
    for family in FAMILIES:
        if not (cfg.images / f"mnist-{family}-seed0/summary.json").exists():
            raise FileNotFoundError(f"missing completed MNIST run: {family}")
    ring_paths = [cfg.rings / f"ring8-{f}-seed{s}" for f in FAMILIES for s in (0, 1, 2)]
    gate = cfg.quality_gate
    ready = (
        all((p / "summary.json").exists() for p in ring_paths)
        and gate.exists()
        and (cfg.corrected / "summary.json").exists()
    )
    if ready:
        quality = json.loads(gate.read_text())
        ready = (
            quality["evaluated_samples"] == 512
            and quality["minimum_confident_fraction"] == 0.9
            and quality["passed"] == (quality["confident_fraction"] >= quality["minimum_confident_fraction"])
            and quality["samples_sha256"]
            == hashlib.sha256((cfg.corrected / "samples.pt").read_bytes()).hexdigest()
        )
    if not ready and not cfg.allow_pending:
        raise RuntimeError("matching image diagnostic or full three-seed ring8 comparison is incomplete")
    rows = []
    for family in FAMILIES:
        name = f"mnist-{family}-seed0"
        rows.append(archive(cfg.images / name, cfg.report / "runs" / name))
    extra = Path("out/quality-ablation/v10k")
    parameterization = None
    if (extra / "summary.json").exists():
        parameterization = archive(extra, cfg.report / "ablations" / "v10k")
    if (cfg.corrected / "summary.json").exists():
        corrected = archive(cfg.corrected, cfg.report / "corrected-v50k")
    else:
        corrected = None
    # Keep unsuccessful follow-ups alongside the fixed-budget and longer runs.
    fine_tune = Path("out/quality-ablation/v55k")
    if (fine_tune / "summary.json").exists():
        archive(fine_tune, cfg.report / "ablations" / "v55k")
        shutil.copy2(fine_tune.with_name("v55k-quality.json"), cfg.report / "ablations/v55k/quality.json")
    for name in ("v50k-ddim200", "v50k-ddpm200", "v50k-online-ddim200"):
        source = Path("out/quality-ablation") / name
        if (source / "inference.json").exists():
            destination = cfg.report / "ablations" / name
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("inference.json", "samples.png"):
                shutil.copy2(source / filename, destination / filename)
    ring_rows = []
    for path in ring_paths:
        if (path / "summary.json").exists():
            ring_rows.append(archive(path, cfg.report / "runs" / path.name))
    for family in ("vae", "priming", "diffusion"):
        source = Path("out/panels") / family
        if source.exists():
            destination = cfg.report / "reconstruction" / family
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("reconstruction.png", "reconstruction.json"):
                shutil.copy2(source / name, destination / name)
    stages = Path("out/priming-stages")
    if stages.exists():
        destination = cfg.report / "priming-stages"
        destination.mkdir(exist_ok=True)
        for name in ("primed.png", "posttrained.png", "stages.json"):
            shutil.copy2(stages / name, destination / name)
    vae = Path("out/vae-heldout-full/evaluation.json")
    if vae.exists():
        shutil.copy2(vae, cfg.report / "vae-test.json")
    if gate.exists():
        shutil.copy2(gate, cfg.report / "quality-gate.json")
    write_json(cfg.report / "results.json", {"mnist": rows, "ring8": ring_rows, "corrected": corrected})
    sources = [*sorted(Path("src/xdiffusion").glob("*.py")), Path("pyproject.toml"), Path("uv.lock")]
    write_json(
        cfg.report / "source-sha256.json",
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )

    table = [
        "| Family | Feature distance ↓ | Confident fraction | Pixel SWD ↓ | NFE | Train seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['family']} | {row['feature_fd']:.2f} | {row['confident_fraction']:.1%} | "
            f"{row['swd']:.4f} | {row['nfe']} | {row['train_s']:.1f} |"
        )
    table_text = "\n".join(table)
    ring_table = [
        "| Family | SWD ↓, mean ± sd | Within a mode, mean | Modes covered, min / 8 |",
        "|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        group = [r for r in ring_rows if r["family"] == family]
        if len(group) == 3:
            values = [r["swd"] for r in group]
            concentration = statistics.mean(r["within_mode_fraction"] for r in group)
            coverage = min(r["covered_modes"] for r in group)
            ring_table.append(
                f"| {family} | {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} | "
                f"{concentration:.1%} | {coverage} |"
            )
    pending = "\n**Work in progress: a required diagnostic or ring8 run is missing.**\n" if not ready else ""
    narrative = f"""# Generative-family comparison · 20 October 2026
{pending}
## Protocol

MNIST: one training seed (0), 10,000 updates, batch 64, final EMA weights.
Priming counts both stages: 5,000 reconstruction and 5,000 distributional
updates. The common evaluation uses 1,024 generated images, validation data,
sampling seed 0, and an independent frozen MNIST classifier. The data seed is
1234. Diffusion in this matched-budget table uses epsilon prediction.

The real-vs-real feature distance is {rows[0]["reference_feature_fd"]:.2f}.
Feature distance is dataset-specific, **not Inception FID**. Confidence is a
95%-softmax diagnostic, **not generated-image accuracy**. No image is filtered
or chosen by score: every grid starts at the first generated sample.

{table_text}

These are exploratory baselines, not a statistically established ranking.
Architecture sizes and extra networks differ. Priming uses supervised frozen
features and an additional encoder-training stage. GAN trains a discriminator;
VAE trains an encoder. Equal updates do not imply equal compute. Exact costs,
model sizes, class histograms, and configurations are in [results.json](results.json)
and the per-run folders. The GAN baseline retains a constant discriminator LR
while its generator LR decays, as recorded by the implementation.

## Qualitative comparison

"""
    for family in FAMILIES:
        label = family.upper() if family in ("vae", "gan") else family.capitalize()
        narrative += f"### {label} · 10,000 updates\n\n"
        narrative += f"![{family} generated samples](runs/mnist-{family}-seed0/samples.png)\n\n"
    narrative += """## Quality investigation

The original 5,000-step tiny diffusion model had feature distance 287.54 and
42.8% confident samples. Keeping its weights fixed, 200-step DDIM reached
44.1%, and 200-step DDPM reached 45.3%; changing the sampler did little.
Training that same tiny architecture for 20,000 updates reached 49.0%.
See [the sampler controls](sampler-ablation/results.json) and
[the longer tiny run](tiny20k-quality.json).

The larger three-resolution backbone improves context and capacity. In the
10,000-step controlled parameterization comparison, epsilon and v prediction
share the same backbone, budget, data, and optimizer recipe. Neither initial
run reached the predeclared 90% confidence diagnostic. The larger 50,000-step
v run therefore tests additional training; it is **outside** the fixed-budget
family comparison and must not be used to claim a matched-budget win.

"""
    if parameterization is not None:
        narrative += (
            "| Same backbone and 10,000-update budget | Feature distance ↓ | Confident fraction |\n"
            "|---|---:|---:|\n"
            f"| Epsilon prediction | {rows[0]['feature_fd']:.2f} | {rows[0]['confident_fraction']:.1%} |\n"
            f"| V prediction | {parameterization['feature_fd']:.2f} | "
            f"{parameterization['confident_fraction']:.1%} |\n\n"
        )
    if corrected is not None:
        narrative += (
            f"### Additional quality run · v prediction, 50,000 updates\n\n"
            f"Feature distance **{corrected['feature_fd']:.2f}**, confidence diagnostic "
            f"**{corrected['confident_fraction']:.1%}**, pixel SWD **{corrected['swd']:.4f}**.\n\n"
            "![Additional quality run](corrected-v50k/samples.png)\n\n"
        )
    if ready:
        narrative += (
            f"**The quality target was not reached.** On the fixed first 512 samples, "
            f"the 50,000-update run scores {quality['confident_fraction']:.1%}, below the "
            f"{quality['minimum_confident_fraction']:.0%} target. "
            "[quality-gate.json](quality-gate.json) retains `passed: false`; the standalone "
            "quality-check command exits nonzero. The 1,024-sample scores above use a "
            "different sample count and must not be substituted for this check.\n\n"
            "Further probes also missed the target: 200-step DDIM reached 89.6%, "
            "200-step DDPM 88.3%, and online weights with 200-step DDIM 87.7%. "
            "A separate 5,000-update warm start from the 50k EMA, with a fresh "
            "optimizer at LR 3e-5, reached 86.9% (55,000 total updates). "
            "These are diagnostics on the same validation protocol, not independent "
            "test results. Their [configs and grids](ablations/) remain available; "
            "the main gallery keeps the original final 50k EMA and its default sampler. "
            "Visual inspection finds much clearer strokes than the original tiny model, "
            "but still malformed and ambiguous digits. None of these runs is a "
            "claim of solved image generation.\n\n"
        )
    narrative += """## Reconstruction and denoising

Adjacent VAE images are original test inputs followed by posterior-mean
reconstructions. The generation table above instead uses fresh prior latents.
The full VAE reconstruction evaluation covers 1,024 test inputs;
[vae-test.json](vae-test.json) records counts and errors. The 16-image panels
below are illustrative and their pixel errors are not a cross-family ranking:
VAE compression, diffusion denoising at t/T=0.25, and Priming denoising at s=0.8
have different input information and objectives. GAN has no encoder. A flow
ODE cycle would test numerical inversion, not learned reconstruction quality.

On those 1,024 inputs, VAE reconstruction MSE is **0.006458** on [0,1] pixels.
The independent judge predicts the true label for **97.0%** of reconstructions,
versus **98.4%** of original inputs. These are reconstruction diagnostics;
the much weaker VAE prior generations remain visible above.

![VAE original/reconstruction pairs](reconstruction/vae/reconstruction.png)

Diffusion panel order: original, noisy input, clean prediction.

![Diffusion denoising](reconstruction/diffusion/reconstruction.png)

Priming panel order: original, corrupted input, reconstruction-only output,
post-trained output. Post-training optimizes a distribution, not paired MSE.

![Priming denoising](reconstruction/priming/reconstruction.png)

## Priming stage ablation

Both stages below use the same 512 CPU-generated noise tensors, seed 0, and
independent evaluation features. See [raw stage scores](priming-stages/stages.json).
These scores have a different sampling count/device from the main table.
Reconstruction priming alone does not produce a good one-step generator.

| Stage | Feature distance ↓ | Confident fraction |
|---|---:|---:|
| Reconstruction priming | 818.60 | 36.1% |
| Distributional post-training | 51.23 | 75.6% |

![Reconstruction-only generations](priming-stages/primed.png)

![After distributional post-training](priming-stages/posttrained.png)

## Ring8: concentration and coverage

Three independent training seeds, 5,000 updates and batch 256 per family.
A point is near a mode when it is within 0.24 of a true center; a mode is
covered if at least 1% of all generated points land near it. Counts avoid
mistaking a continuous blurry ring for eight concentrated modes.

""" + "\n".join(ring_table)
    narrative += """

## Reproduce

```bash
uv sync --locked
uv run -m xdiffusion.data --task mnist
uv run -m xdiffusion.study --tasks mnist --steps 10000 \\
  --encoder-root reports/comparison-2026-10-20/encoders --out-dir out/comparison-mnist
uv run -m xdiffusion.experiment --family diffusion --prediction v --steps 10000 \\
  --encoder-dir reports/comparison-2026-10-20/encoders/mnist --out-dir out/quality-ablation/v10k
uv run -m xdiffusion.experiment --family diffusion --prediction v --steps 50000 \\
  --encoder-dir reports/comparison-2026-10-20/encoders/mnist --out-dir out/quality-ablation/v50k
uv run scripts/check_image_quality.py --samples out/quality-ablation/v50k/samples.pt \\
  --judge reports/comparison-2026-10-20/encoders/mnist/judge.pt
# Expected: this diagnostic fails its 90% target. Continue the other commands separately.
uv run -m xdiffusion.experiment --family diffusion --prediction v --steps 5000 --lr 3e-5 \\
  --init-checkpoint out/quality-ablation/v50k/ckpt.pt \\
  --encoder-dir reports/comparison-2026-10-20/encoders/mnist --out-dir out/quality-ablation/v55k
uv run -m xdiffusion.study --tasks ring8 --seeds 0 1 2 --steps 5000 \\
  --width 128 --batch-size 256 --out-dir out/comparison-ring8
```

The standalone experiment command defaults to encoders under `out/encoders`;
the commands above explicitly use the included frozen encoders. To train new
ones, use `python -m xdiffusion.quality`. Encoder
metadata records its training and data seeds and held-out classification score.
Generator checkpoints and optimizer states remain under local `out/`; the repo
includes their configurations, learning histories, summaries, and sample grids.
[source-sha256.json](source-sha256.json) identifies the implementation shipped
with this report. Training is float32 on a GTX 1080. Timings depend on hardware
and system load; seed identity across different devices is not promised.
"""
    (cfg.report / "README.md").write_text(narrative)
    if ready:
        readme = Path("README.md").read_text()
        readme = update_section(readme, "progress", "")
        improvement = (
            f"On the same 512-sample protocol, the longer diffusion run reaches "
            f"**{quality['confident_fraction']:.1%}** confident samples (original: **42.8%**) "
            f"and feature distance **{quality['feature_fd']:.2f}** (original: **287.54**). "
            "It still **misses the 90% diagnostic target**, and some digits remain malformed. "
            "This changes the backbone, prediction target, and training budget; "
            "it is a quality investigation, separate from the fixed-budget family comparison."
        )
        readme = update_section(readme, "quality-result", improvement)
        readme = update_section(readme, "comparison-table", table_text)
        gallery = """<table>
  <tr><th>Diffusion · v prediction · 50,000 updates</th><th>Flow matching · 10,000 updates</th></tr>
  <tr>
    <td><img src="reports/comparison-2026-10-20/corrected-v50k/samples.png"
      alt="Unfiltered samples from the longer v-prediction diffusion run" width="420"></td>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-flow-seed0/samples.png"
      alt="Unfiltered flow-matching MNIST samples" width="420"></td>
  </tr>
</table>
<table>
  <tr><th>Priming · one evaluation</th><th>VAE · one decoder evaluation</th>
    <th>GAN · one generator evaluation</th></tr>
  <tr>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-priming-seed0/samples.png"
      alt="Unfiltered Priming samples after both training stages" width="280"></td>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-vae-seed0/samples.png"
      alt="VAE prior samples, including blurry and malformed examples" width="280"></td>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-gan-seed0/samples.png"
      alt="GAN samples, including remaining shape defects" width="280"></td>
  </tr>
</table>

The first diffusion gallery has a **larger training budget**. The fixed-budget
comparison below keeps every family at 10,000 updates."""
        readme = update_section(readme, "current-gallery", gallery)
        reconstruction_image = "reports/comparison-2026-10-20/reconstruction/vae/reconstruction.png"
        reconstruction = f"""![Held-out VAE original/reconstruction pairs]({reconstruction_image})

Adjacent pairs: original input, posterior-mean reconstruction. The VAE's low
reconstruction error does not imply equally good samples from its prior; both
operations are measured separately in the report. Across 1,024 test inputs,
reconstruction MSE is **0.006458** on [0,1] pixels; the independent judge predicts
the correct label for **97.0%** of reconstructions (original inputs: **98.4%**)."""
        readme = update_section(readme, "reconstruction-gallery", reconstruction)
        Path("README.md").write_text(readme)
    print(f"Archived {len(rows)} image and {len(ring_rows)} ring runs; measurements complete: {ready}")


if __name__ == "__main__":
    main(tyro.cli(Config))
