# From points to pixels

This companion starts with the epsilon-prediction baseline. The
[family guide](families.md) covers v prediction, flow matching, Priming, VAE,
GAN, and the newer comparison protocol.

A diffusion model learns to remove noise. The data can be a pair of coordinates
or thousands of pixel intensities; the forward process and training objective
are the same. Only the denoiser changes.

## 1. Add known noise

Let x₀ be clean data, ε a standard normal tensor of the same shape, and t an
integer sampled uniformly from 1 through T. Construct

    xₜ = sqrt(ᾱₜ) x₀ + sqrt(1 − ᾱₜ) ε.

ᾱₜ is the fraction of signal power left at time t. It starts at 1 for clean
data and falls almost to zero. The cosine schedule in `diffusion.py` defines
this fall, discretizes it into per-step βₜ, caps βₜ at 0.999, then recomputes
ᾱₜ as the cumulative product of (1 − βₜ). This avoids an exactly zero divisor.

We use an explicit clean index 0, unlike implementations that store their
first noisy level at index 0. Noise levels passed to the network are normalized
to t/T. Sinusoidal time features internally rescale that value by 1000.

## 2. Predict the noise

The network receives xₜ and t/T, then predicts ε. The loss is the mean squared
error between known and predicted noise, averaged over examples and features.
This is the simple noise-prediction objective, not the full variational bound.

For points, an MLP concatenates coordinates and time features, then runs
residual hidden blocks. For images, a U-Net adds projected time features to
convolutional residual blocks. The current image backbone visits full, half,
and quarter resolution, then upsamples with saved skip features. Its bottleneck
has context over a complete MNIST image. The original two-resolution network
is retained as `model.architecture=tiny` for the old baselines. Both return
the input's shape.

The point MLP blends its learned output with the input: `t*x + (1-t)*network`.
At t=1 the input is almost pure noise, so epsilon ≈ x. Enforcing this identity
boundary prevents a small error at the final cosine level from being magnified
by division by sqrt(ᾱT). This is a deliberate approximation at a level whose
signal power is already tiny, and keeps unbounded point sampling stable without
clipping coordinates. The image U-Net uses the ordinary output and clips x̂₀.

Training uses AdamW, linear learning-rate warmup followed by cosine decay to
10% of the peak rate, gradient clipping, and exponential moving averages of
the parameters. The EMA decay ramps up early so random initial weights do not
linger. Setting `--ema-decay 0` makes the evaluated weights equal current weights.

## 3. Turn noise into data

At inference start from xT ~ N(0, I), then apply the denoiser repeatedly.
From a predicted ε, estimate clean data:

    x̂₀ = (xₜ − sqrt(1 − ᾱₜ) ε̂) / sqrt(ᾱₜ).

For images we clamp this estimate to the data range [-1, 1]. We leave points
unclamped because some distributions have unbounded support.

For deterministic DDIM, choose an earlier level s and set

    xₛ = sqrt(ᾱₛ) x̂₀ + sqrt(1 − ᾱₛ) ε̂.

After clipping x̂₀ we recompute ε̂ so the decomposition of xₜ remains consistent.
Choosing s several levels earlier skips network evaluations. The last level
is s=0, so the final result is the estimated clean data. Deterministic means
the initial noise determines the entire path; different initial seeds still
produce different samples.

DDPM instead visits every level. With αₜ = ᾱₜ / ᾱₜ₋₁ and βₜ = 1 − αₜ,
its posterior mean and variance are

    μ = [sqrt(ᾱₜ₋₁) βₜ x̂₀ + sqrt(αₜ) (1 − ᾱₜ₋₁) xₜ] / (1 − ᾱₜ)
    σ² = βₜ (1 − ᾱₜ₋₁) / (1 − ᾱₜ).

Sample xₜ₋₁ = μ + σz, except at the final transition where σ²=0. Skipping
levels with these adjacent-step formulas would change the algorithm, so the
CLI rejects shortened DDPM runs.

These follow [DDPM](https://arxiv.org/abs/2006.11239), the cosine schedule in
[Improved DDPM](https://arxiv.org/abs/2102.09672), and
[DDIM](https://arxiv.org/abs/2010.02502). This repo implements fixed posterior
variance and deterministic DDIM (η=0); it does not implement learned variance.

## 4. Separate learning from measurement

Training samples random minibatches from a fixed training set. Model seeds
change initialization and training randomness. The independent data seed
fixes train/validation/test membership across model seeds. Validation uses its
own generator for time and noise, so evaluating more often does not consume
the training random stream. CUDA kernels may still be nondeterministic.

Checkpoint selection asks whether EMA weights predict validation noise well.
Sampling quality asks whether generated samples resemble held-out data. These
questions differ; a lower noise MSE need not produce the best-looking grid.

SWD projects both equal-size sample sets onto fixed unit directions. In one
dimension, sorting provides the optimal matching for Wasserstein-1 distance.
Average the absolute differences of matched values, then average over
directions. The comparison against a second held-out subset estimates the
scale of finite-sample variation; it is not a strict lower bound on every run.
The Gaussian baseline shows what the sampler starts from before learning.

Use the same task, split, data seed, sample count, sampler, reverse-step count,
and batch size when comparing runs. Do not pool metrics from different
settings. Default multi-seed presets hold these settings fixed within a task.
The table printer assumes you supply compatible runs.

## 5. Increase difficulty one change at a time

Start with `gaussian`: inspect the learned center, scale, and elongation. Move
to `ring8`: inspect whether all eight modes appear. Try `moons` for curved
geometry, then `checkerboard` for boundaries. In each case compare the generated
scatter plot against the reference data, not just a scalar loss.

`shapes` changes the denoiser to a U-Net while keeping data generation offline
and interpretable. MNIST and Fashion-MNIST introduce real images. CIFAR-10 adds
three channels, color, texture, and much more varied content. Its longer preset
is an experiment budget, not evidence that this tiny U-Net matches research
systems. Pixel SWD cannot establish that. Add a well-specified perceptual metric
before making serious natural-image quality comparisons.

To add a task, add its loader in `data.py` and its budget in `benchmark.py`.
To try a different denoiser, preserve `(x, time) -> noise` and adjust
`make_model`. Keep the objective, splits, and evaluation settings fixed so
the comparison answers the architectural question you intended to ask.
