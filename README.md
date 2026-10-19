# xDiffusion

Small, readable diffusion models in PyTorch, in the spirit of nanoGPT and the
neighboring xByteGPT repository. Learn the same noise-prediction algorithm on
2D points, generated images, and real image datasets.

**PyTorch · jaxtyping · einops · tyro.** Plain modules and dataclasses, one
training loop, no trainer framework, registries, or configuration inheritance.

## Quickstart

```bash
uv sync
uv run -m xdiffusion.train                         # eight Gaussian modes, ~3000 steps
uv run -m xdiffusion.sample --checkpoint out/ring8/ckpt.pt
uv run -m xdiffusion.evaluate --checkpoint out/ring8/ckpt.pt  # held-out test data
uv run pytest
uv run ruff check .
```

Every command takes `--help`. Dataclass fields become flags such as
`--model.width 64` or `--diffusion.steps 100`. Use a fresh `--out-dir` for each
run; existing runs are protected from accidental overwrite.

[Measured runs and sample plots](reports/baseline-2026-10-19/README.md) include
three seeds on each point task, short image runs, and explicit limits on what
those runs establish.

For a quick CPU check:

```bash
uv run -m xdiffusion.train --device cpu --steps 20 --model.width 32 \
  --sampling.samples 64 --out-dir out/smoke
```

## Benchmark curriculum

| Task | Data | What it exercises | Denoiser | Preset steps |
|---|---|---|---|---:|
| `gaussian` | translated anisotropic 2D normal | mean and covariance | MLP | 1,500 |
| `ring8` | eight narrow Gaussian modes | mode coverage and concentration | MLP | 3,000 |
| `moons` | two noisy curved manifolds | nonlinear geometry | MLP | 3,000 |
| `checkerboard` | eight occupied squares | sharp boundaries and empty regions | MLP | 5,000 |
| `shapes` | 16×16 circles and squares | spatial structure, translation, size | U-Net | 5,000 |
| `mnist` | 28×28 handwritten digits | real multimodal grayscale images | U-Net | 20,000 |
| `fashion` | 28×28 clothing | texture and more varied silhouettes | U-Net | 30,000 |
| `cifar10` | 32×32 color photographs | natural images and color | U-Net | 100,000 |

Difficulty is a teaching progression, not a measured ranking. Presets are
starting budgets, not equal-compute experiments or published quality targets.
CIFAR-10 is intentionally a harder ceiling for this small attention-free U-Net.

```bash
# Offline suite: four point distributions + generated shapes, three model seeds.
uv run -m xdiffusion.benchmark

# Shorter point-only comparison.
uv run -m xdiffusion.benchmark --tasks gaussian ring8 moons checkerboard \
  --seeds 0 1 2 --steps 1000 --out-dir out/points

# Download once. Training never downloads data implicitly.
uv run -m xdiffusion.data --task mnist
uv run -m xdiffusion.data --task fashion
uv run -m xdiffusion.data --task cifar10

uv run -m xdiffusion.train --data.task mnist --model.width 32 \
  --batch-size 64 --lr 3e-4 --steps 20000 --out-dir out/mnist
uv run -m xdiffusion.benchmark --tasks mnist fashion cifar10 --seeds 0 \
  --out-dir out/images

uv run -m xdiffusion.compare out/points/ring8-seed0 out/points/ring8-seed1
```

`benchmark.py` contains the complete preset table. Synthetic train/val/test
sets use independent fixed seeds. Real images reserve 5,000 examples from the
official training set for validation, using a fixed permutation. The official
test set is used only by `evaluate --split test` (the evaluation CLI default).
Labels are ignored: all models are unconditional. Image pixels are in [-1, 1];
2D coordinates retain their natural scale. There is no augmentation.

## Read the code

```text
src/xdiffusion/
  data.py        synthetic distributions, torchvision splits, download CLI
  models.py      sinusoidal time features, residual MLP, small U-Net
  diffusion.py   cosine noise schedule, noise MSE, DDPM and DDIM sampling
  train.py       AdamW, warmup/cosine LR, gradient clipping, EMA, validation
  sample.py      bounded-memory sampling and checkpoint-to-plot CLI
  metrics.py     fixed-noise validation and sliced Wasserstein distance
  evaluate.py    held-out scoring, reference baselines, generation timing
  benchmark.py   task presets and sequential multi-seed runs
  compare.py     per-run table and mean ± standard deviation
  common.py      device, checkpoints, JSON, sample plots
tests/           equations, shapes, splits, metrics, end-to-end runs
doc/companion.md the algorithm and measurement explained
```

Both models implement one contract:

```python
model(noisy: Float[Tensor, "batch *shape"],
      time: Float[Tensor, "batch"]) -> Float[Tensor, "batch *shape"]
```

`time` is t/T. The model predicts the Gaussian noise added to clean data.
`shape` is `(2,)` for points or `(channels, height, width)` for images.
`make_model` chooses the MLP or U-Net from that shape. Read
[the companion](doc/companion.md) for the equations and design decisions.

Tensor-facing functions carry jaxtyping annotations. Pytest checks shapes and
dtypes at runtime through jaxtyping + beartype; training pays no annotation
checking cost. Einops names axes when rearranging data. Tyro exposes nested
dataclass configuration without config files or `exec`.

## Measurements and artifacts

Each training run writes:

- `config.json`: full settings, including independent data and model seeds.
- `history.json`: fixed validation noise MSE and accumulated training time.
- `ckpt.pt`: EMA weights with the lowest validation noise MSE, model/data/noise
  configs, data shape, and training step. Inference checkpoint, without optimizer
  state; training resumption is not implemented.
- `summary.json`: validation scores, parameter count, training examples/s,
  generation samples/s, sampler settings, Python/PyTorch versions and hardware.
- `samples.pt` and `samples.png`: generated points or image grid from that checkpoint.

**Noise MSE** measures the denoising objective on fixed validation data, times,
and noise. It is not a likelihood or a direct sample-quality score. Checkpoint
selection uses this metric; final generation uses the chosen EMA checkpoint.

**SWD** is sliced Wasserstein-1 distance: project generated and held-out data
onto 128 fixed random unit directions, sort, and average absolute differences.
Lower is better within the same task and evaluation settings. `reference_swd`
compares two disjoint held-out subsets to expose finite-sample variation.
`gaussian_swd` scores an untrained standard-normal generator as a baseline.
All three use equal sample counts and the same projection directions.

For images SWD runs in **pixel space**. It is a cheap diagnostic, not FID or a
measure of perceptual quality. Inspect sample grids. Scores across different
datasets are not comparable; even image SWD can miss semantic failures.

Training timing includes batch selection, transfers, forward/backward, AdamW,
EMA, and logging; it excludes validation and checkpoint writes. Generation is
warmed up, synchronized on CUDA, and includes copying samples to CPU. Compare
speed only on the same machine, batch size, and sampler settings. Seeds repeat
within a given environment; bitwise cross-device reproducibility is not promised.

## Sampling choices

The default is deterministic DDIM with 50 reverse steps. The initial Gaussian
still depends on the sampling seed. DDPM uses every forward noise level and
adds posterior noise on all but the final transition:

```bash
uv run -m xdiffusion.sample --checkpoint out/ring8/ckpt.pt \
  --sampling.sampler ddpm --sampling.steps None --out-dir out/ddpm
uv run -m xdiffusion.sample --checkpoint out/ring8/ckpt.pt \
  --sampling.steps 20 --out-dir out/ddim20
```

Image predictions are clipped to [-1, 1] during sampling; point predictions
are not clipped. Sampling steps must be <= the checkpoint's diffusion steps.

## Scope

Single-device float32 training on CPU or CUDA. No DDP, latent autoencoder,
classifier-free guidance, learned variance, or mixed-precision machinery.
The U-Net has two spatial resolutions, residual blocks with time conditioning,
GroupNorm, and a skip connection. `model.depth` controls only the MLP.

Python 3.12–3.13; the lockfile pins dependencies. PyTorch 2.7.1 and torchvision
0.22.1 are paired deliberately, including CUDA support for this machine's GTX
1080 (Pascal). `uv sync` uses the standard PyPI wheels.

## References

- [DDPM — Ho et al.](https://arxiv.org/abs/2006.11239): noise-prediction training and reverse diffusion.
- [Improved DDPM — Nichol & Dhariwal](https://arxiv.org/abs/2102.09672): cosine noise schedule.
- [DDIM — Song et al.](https://arxiv.org/abs/2010.02502): deterministic sampling with skipped noise levels.
