# Running and inspecting experiments

Use `python -m xdiffusion.train` for all five families and
`python -m xdiffusion.generate` for every checkpoint. `study` runs the same
trainer repeatedly across families, tasks, and seeds. `benchmark` supplies
starting budgets for a diffusion-only curriculum. Neither has a separate trainer.

## Training settings

The default single run is diffusion on ring8: 3,000 updates, batch 256, width 128,
LR 1e-3. The study defaults to MNIST with width 32, batch 64 and LR 3e-4; point
studies use LR 1e-3 and width at least 64. These are starting recipes.

All optimizers are AdamW. Effective arguments are stored under `optimizer_recipe`
in checkpoints and summaries:

| Optimizer | Betas | Weight decay | LR schedule |
|---|---|---:|---|
| Diffusion, flow, VAE, Priming | (0.9, 0.999) | 1e-4 | Stage-local warmup and cosine |
| GAN generator | (0.5, 0.999) | 1e-4 | Warmup and cosine |
| GAN discriminator | (0.5, 0.999) | 0.01 | Constant |

These preserve the historical comparison recipe, including its discriminator
asymmetry. They are explicit baselines, not recommendations for optimal settings.
The generator's warmup lasts up to 100 updates and its cosine envelope ends at
10% of peak LR. Priming starts a fresh optimizer at the stage transition, with
`--post-lr 1e-4`. Gradient norms are clipped to 1.

Override `--lr`, `--betas`, and `--weight-decay` for the generator. GAN also accepts
`--discriminator-lr`, `--discriminator-betas`, `--discriminator-weight-decay`,
and `--discriminator-schedule constant|cosine`. An omitted discriminator LR
inherits the generator's peak LR. Its cosine option uses the generator's warmup
and decay envelope. Changing these settings creates a new recipe and requires
new measurements; earlier report numbers remain tied to their original settings.

## Histories and checkpoints

Every run writes:

| File | Contents |
|---|---|
| `config.json` | User configuration |
| `history.json` | Instantaneous objectives and diagnostics at log intervals |
| `source-sha256.json` | Source and available dependency-file hashes at training start |
| `summary.json` | Budget, effective optimizer recipes, model sizes, timing, sample metrics |
| `samples.pt`, `samples.png` | Full generated tensor and its first images, or point plot |
| `ckpt.pt` | Final EMA plus full training state |
| `last.pt` | Latest resumable training state |
| `primed.pt` | Priming's reconstruction-stage state, when applicable |

All families log the effective generator LR and gradient norm before clipping.
Diffusion logs prediction MSE; flow logs velocity MSE. Priming logs reconstruction
MSE or its distributional MMD surrogate, with the stage identified. VAE logs summed
reconstruction negative log likelihood and latent KL before beta weighting.
GAN logs discriminator logistic loss, weighted lazy R1 penalty, generator loss,
real/fake logits, discriminator gradient norm, and discriminator LR. `r1_applied`
marks penalty updates; a zero penalty on other steps is expected. These are
instantaneous diagnostics, not averages over the log interval. Objective values
have different units across families and are not a ranking.

`--resume` restores online weights, EMA, optimizers, private random streams, and
history from `last.pt`. It requires the same configuration and total budget.
Changing the budget would change the LR schedule. Reproducibility also depends
on the source, dependencies, device, and environment.

For a new diffusion or flow fine-tuning phase, use a fresh output directory and
`--init-checkpoint path/to/ckpt.pt`. This loads the parent EMA with a fresh
optimizer. Summaries record `parent_steps`, `total_steps`, and `total_train_s`.
It is a separate phase, not an extension of the original fixed-budget result.

## Sampling and evaluation

`generate` loads the saved family and data settings, draws fresh noise, and scores
against the official test split by default. Synthetic test data use an independent
seed. Use `--split val` to reproduce validation measurements and `--no-evaluate`
to sample without loading any reference data. VAE evaluation also writes
posterior-mean input/reconstruction pairs and reconstruction errors.

```bash
uv run -m xdiffusion.generate --checkpoint out/ring8/ckpt.pt \
  --samples 1024 --seed 0 --split test --out-dir out/test

# Diffusion sampler override; DDPM requires all forward noise levels.
uv run -m xdiffusion.generate --checkpoint out/ring8/ckpt.pt \
  --sampler ddpm --sampling-steps 200 --out-dir out/ddpm-test
```

Compare runs using the same split, reference count, sampling seed, batch size,
and solver budget. Classifier confidence is a diagnostic; inspecting samples
and coverage remains necessary. Frozen MNIST/Fashion encoders must match the
data task and split seed. Training and studies validate their metadata before
spending compute. To train new encoders, run `python -m xdiffusion.quality`.

## Reports and historical compatibility

Create a new report directory from completed runs:

```bash
uv run scripts/build_comparison_report.py \
  --runs out/compare-ring8/ring8-diffusion-seed0 out/compare-ring8/ring8-flow-seed0 \
  --report reports/my-ring-study
```

The exporter copies measured configs, summaries, histories, plots, and available
training-source hashes, then renders per-run tables. It refuses existing report
directories. It does not edit the main README or invent source provenance for
older runs. Full checkpoints and sample tensors stay in the run directories.

The [original diffusion report](../reports/baseline-2026-10-19/README.md) and
[family comparison](../reports/comparison-2026-10-20/README.md) give exact revisions
for reproducing their historical settings. Their artifacts are preserved.

The checkpoint loader accepts original diffusion checkpoints and the former
comparison format. Original checkpoints without an architecture field select
the original tiny U-Net. Model-only legacy checkpoints support sampling with
`--no-evaluate`; scoring requires saved data metadata. The new schema also records
effective optimizer recipes and source hashes.

`experiment` remains an alias for `train`; `sample` and `evaluate` retain their
older diffusion sampling flags as adapters to `generate`. The former training
flags `--diffusion.steps`, `--diffusion.prediction`, `--sampling.samples`, and
`--sampling.steps` become `--diffusion-steps`, `--prediction`, `--samples`, and
`--sampling-steps`. The consolidated trainer always evaluates final EMA weights.
To reproduce the former best-validation-noise-MSE selection, use the pinned
historical revision rather than the current trainer.
