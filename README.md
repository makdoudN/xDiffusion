# xDiffusion

### Small generative models you can read, run, and compare.

**Diffusion · flow matching · Priming Models · VAE · GAN.**
Plain PyTorch, jaxtyping tensor contracts, einops, and tyro dataclass CLIs.

[![Checks](https://github.com/makdoudN/xDiffusion/actions/workflows/check.yml/badge.svg)](https://github.com/makdoudN/xDiffusion/actions/workflows/check.yml)
[Get started](#get-started) · [Compare](#compare-the-models) · [Read the code](#read-the-code) · [The math](doc/families.md)

<table>
  <tr><th>Diffusion</th><th>Flow matching</th><th>Priming</th></tr>
  <tr>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-diffusion-seed0/samples.png" alt="Unfiltered diffusion generations" width="280"></td>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-flow-seed0/samples.png" alt="Unfiltered flow generations" width="280"></td>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-priming-seed0/samples.png" alt="Unfiltered Priming generations" width="280"></td>
  </tr>
  <tr><th>VAE prior</th><th>GAN</th><th>Same update budget</th></tr>
  <tr>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-vae-seed0/samples.png" alt="VAE prior generations, including blur" width="280"></td>
    <td><img src="reports/comparison-2026-10-20/runs/mnist-gan-seed0/samples.png" alt="GAN generations, including shape defects" width="280"></td>
    <td>MNIST · 10,000 updates · batch 64 · seed 0.<br>First generated images, without filtering.</td>
  </tr>
</table>

These are small educational baselines. Architecture sizes and training resources
differ; the [measured report](reports/comparison-2026-10-20/README.md) records their
recipes, scores, and remaining defects. Its results are pinned to the original code revision.

## Why this repo?

Start with a scatter plot where missing modes are visible, then move to images
where spatial structure matters. Compare the equations, optimization, and
sampling costs directly. In the spirit of [xByteGPT](https://github.com/makdoudN/xByteGPT)
and [nanoGPT](https://github.com/karpathy/nanoGPT), the code uses ordinary functions
and an explicit training loop, with each model's objective in its own small module.

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/makdoudN/xDiffusion.git
cd xDiffusion
uv sync --locked

# Diffusion on eight Gaussian modes; no downloads needed.
uv run -m xdiffusion.train

# Any family checkpoint uses the same generation and evaluation command.
uv run -m xdiffusion.generate --checkpoint out/ring8/ckpt.pt
```

Open `out/ring8/samples.png`: orange points are generated; blue points are
held-out references. Training uses CUDA when available and otherwise CPU.
Python 3.12–3.13; the locked PyTorch/torchvision pair supports the GTX 1080.

```bash
# A quick CPU installation check, not a trained-model result.
uv run -m xdiffusion.train --device cpu --steps 20 --model.width 32 \
  --samples 64 --out-dir out/smoke

# Choose a family and a fresh directory.
uv run -m xdiffusion.train --family flow --out-dir out/flow-ring8

# Generate without loading or scoring a reference dataset.
uv run -m xdiffusion.generate --checkpoint out/flow-ring8/ckpt.pt --no-evaluate
```

Every command accepts `--help`. Training protects existing run directories.
`ckpt.pt` contains final EMA weights; `last.pt` also stores optimizer and random
states. `--resume` continues the same configured budget. See
[workflows and compatibility](doc/workflows.md) for artifact details and fine-tuning.

## Compare the models

| Family | Training signal | Default generation cost |
|---|---|---:|
| Diffusion | Gaussian noise prediction; epsilon or v target | 50 DDIM evaluations |
| Flow matching | Velocity along straight Gaussian-to-data paths | 25 Heun steps = 50 evaluations |
| Priming | Fixed-noise reconstruction, then feature distribution matching | 1 evaluation |
| VAE | Reconstruction likelihood + latent KL | 1 decoder evaluation |
| GAN | Non-saturating logistic adversarial loss + lazy R1 | 1 generator evaluation |

```bash
# All five families, three training seeds, synthetic data.
uv run -m xdiffusion.study --tasks ring8 --seeds 0 1 2 --steps 5000 \
  --width 128 --batch-size 256 --out-dir out/compare-ring8

# Real images: explicit download, then use the included frozen encoders.
uv run -m xdiffusion.data --task mnist
uv run -m xdiffusion.study --tasks mnist --steps 10000 \
  --encoder-root reports/comparison-2026-10-20/encoders --out-dir out/compare-mnist
```

Equal updates do not imply equal compute. Priming uses an additional supervised
training encoder; evaluation uses independently trained weights. This is a small
adaptation of [Priming Models](https://arxiv.org/abs/2610.20331), using the
Laplacian-MMD alternative. It is not a reproduction of the released ImageNet model.
The historical MNIST comparison has one training seed, so its ranking is exploratory.

New histories expose VAE reconstruction/KL, GAN generator/discriminator losses
and R1, gradient norms, and actual learning rates. Effective AdamW settings are
saved explicitly. [The workflow guide](doc/workflows.md) explains these diagnostics
and how to archive a new study without overwriting earlier evidence.

### Generation and reconstruction answer different questions

![VAE test originals and posterior-mean reconstructions](reports/comparison-2026-10-20/reconstruction/vae/reconstruction.png)

Adjacent pairs: original test input, VAE posterior-mean reconstruction. The gallery
at the top instead decodes fresh prior latents. Low reconstruction error does not
ensure good prior generations. `generate` saves both operations for VAE checkpoints.

The [quality investigation](reports/comparison-2026-10-20/README.md#quality-investigation)
contains the longer diffusion run, failed quality diagnostic, and sampler controls.
Classifier confidence and dataset-specific feature distance are useful diagnostics;
inspect the images too. The feature score is not Inception FID.

## From points to pixels

| Task | What to inspect |
|---|---|
| `gaussian` | Center, scale, covariance |
| `ring8` | Eight concentrated modes and their coverage |
| `moons` | Curved manifolds |
| `checkerboard` | Sharp boundaries and empty regions |
| `shapes` | Position, size, and brightness in 16×16 images |
| `mnist` | Handwritten digits |
| `fashion` | Clothing silhouettes and texture |
| `cifar10` | Color and natural-image structure; no measured result included |

Synthetic tasks stay offline. Real datasets require an explicit `xdiffusion.data`
download. Small feature encoders support MNIST and Fashion-MNIST. `benchmark`
runs diffusion-only task presets through the same trainer; `study` compares families.

## Read the code

```text
src/xdiffusion/
  diffusion.py   Cosine noise, epsilon/v targets, DDPM and DDIM
  flow.py        Straight-path matching, Euler and Heun integration
  priming.py     Reconstruction priming and frozen-feature MMD
  vae.py         Encoder, decoder, reconstruction and KL
  gan.py         Discriminator and logistic objectives
  models.py      Time features, point MLP, two U-Net sizes
  run.py         Config, optimizer recipes, checkpoint loading, generation, scores
  train.py       One training loop, EMA, diagnostics, exact-budget resume
  generate.py    Samples, held-out evaluation, VAE reconstruction pairs
  study.py       Family comparisons across tasks and seeds
  data.py        Synthetic generators, image splits, explicit downloads
```

Read [the diffusion companion](doc/companion.md), then [the family guide](doc/families.md).
Older command names are compatibility adapters described in [workflows](doc/workflows.md).

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests check equations, gradients, tensor contracts, splits, checkpoint compatibility,
and deterministic resume. They do not establish model quality or benchmark fairness.
