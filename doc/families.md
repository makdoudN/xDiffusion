# Five generative objectives

The models share data splits, a final-EMA evaluation protocol, and sample plots.
They do not share a scalar training loss. Comparing their objective values would
be meaningless: the quantities, normalization, and optimization problems differ.

## Diffusion

`diffusion.py` implements a cosine forward process with a clean index 0:

    x_t = sqrt(a_t) x + sqrt(1-a_t) z,  z ~ N(0, I).

Epsilon prediction regresses z. The alternative v prediction regresses

    v_t = sqrt(a_t) z - sqrt(1-a_t) x.

For epsilon prediction, estimating clean data divides by sqrt(a_t). Near the
almost-zero-SNR endpoint this magnifies prediction errors. V prediction instead
uses `x_hat = sqrt(a_t) x_t - sqrt(1-a_t) v_hat`, which has no small divisor.
This is a diffusion parameterization; it is different from the straight-path
velocity used by flow matching below.

Both parameterizations support DDPM with posterior variance and deterministic
DDIM with skipped noise levels. The epsilon-only identity boundary on the point
MLP is disabled for v prediction. Checkpoints record their parameterization.
Older checkpoints retain their original epsilon model and tiny architecture.

Sources: [DDPM](https://arxiv.org/abs/2006.11239),
[DDIM](https://arxiv.org/abs/2010.02502),
[v parameterization](https://arxiv.org/abs/2202.00512).

## Flow matching

`flow.py` learns a velocity field along independent Gaussian-to-data paths:

    x_t = (1-t) z + t x,   target velocity = x-z,   t ~ Uniform(0,1).

The loss is mean squared velocity error. Sampling integrates from t=0 to t=1,
starting at fresh Gaussian noise. Euler uses one field evaluation per step;
Heun uses two. The default 25 Heun steps therefore use 50 evaluations, matching
the 50-step DDIM evaluation budget. Their measured runtimes can still differ.

This is conditional flow matching with independent endpoint coupling, not a
normalizing flow implementation with a change-of-variables likelihood API.
The point backbone uses a raw field output, without the epsilon identity gate.

Source: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

## Priming Models

`priming.py` is an unconditional teaching adaptation of
[Are We Overthinking One-Step Generation?](https://arxiv.org/abs/2610.20331).
It follows the two-stage framework, with the paper's Laplacian-MMD alternative
for post-training. It does not reproduce the released class-conditional
ImageNet transformer, its large pretrained encoders, or its OT configuration.

**Stage 1 — reconstruction.** With a fixed s=0.8, train the generator to predict
x from `(1-s)*x + s*z`. This stage has informative corrupted inputs and direct
paired targets. Reconstruction training alone does not establish that the
same network will generate good images from pure noise.

**Stage 2 — distribution matching.** Initialize the same network from its online
pretrained weights, keep its EMA, and start a fresh optimizer. Generate images
from pure Gaussian noise. In frozen feature space, optimize

    D(online, real) - D(online, stop_gradient(negative)),
    D(a,b) = mean[-exp(-||a_i-b_j|| / bandwidth)].

The negative reference is an independent generated batch with no gradient.
Attraction moves outputs toward real samples; the self-reference supplies
repulsion that discourages collapse. This is a stop-gradient surrogate with
the gradient of half the population MMD-squared, not a numerical MMD-squared
estimate. A finite-batch loss value can be negative.

For MNIST and Fashion-MNIST, the frozen training encoder is a small classifier
trained on the training partition. We combine spatial features at two depths
and its semantic embedding. Each block is scaled using training-data feature
variation and the square root of its dimension. The generator receives no class
labels, but these supervised features are an extra resource. Their setup cost
must be considered when interpreting a comparison with the other families.
For synthetic tasks, raw coordinates or pixels provide the feature space.

The evaluation classifier has independently trained weights and a separate
seed. It is never used in generator optimization. The post-training loss has
no diffusion teacher, GAN discriminator, or Jacobian-vector product.

The configured training budget includes both stages: 10,000 updates means
5,000 reconstruction updates and 5,000 distributional updates by default.
`primed.pt` preserves the first stage so both its generations and its denoising
outputs can be inspected alongside the final model.

## Variational autoencoder

`vae.py` has an encoder producing a diagonal Gaussian posterior, a reparameterized
latent sample, and a decoder. A standard-normal prior supplies latent samples
for generation. The image loss is summed pixel binary cross entropy plus beta
(default 1) times the summed Gaussian KL, then averaged over the batch. For
coordinates, reconstruction uses a fixed Gaussian decoder standard deviation
of 0.1 instead. These choices are recorded in the implementation and config.

The image Bernoulli likelihood is a conventional proxy for normalized grayscale
intensities; this small model does not learn a sophisticated likelihood for
natural color photographs. Its decoder has a global latent bottleneck and no
input-to-output skip connection.

Reconstruction panels decode the posterior mean of actual held-out inputs.
Unconditional samples decode fresh standard-normal latents. These are different
operations and are saved separately. Parameter tables distinguish the decoder
used for generation from the full training model, which includes its encoder.

Source: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

## Generative adversarial network

`gan.py` uses a learned discriminator with a non-saturating logistic objective.
Each update includes a discriminator step on real and detached generated data,
then a generator step through the frozen discriminator. A lazy R1 real-input
gradient penalty is applied every 16 updates. Model and discriminator optimizers
are separate; EMA applies to the generator only.

The GAN generator and VAE decoder have the same topology, with independent
weights and different objectives. Neither shares weights with the other models.
The GAN has no encoder, so no direct input reconstruction score is reported.
Its historical discriminator recipe uses constant LR and weight decay 0.01;
its generator uses cosine LR and weight decay 1e-4. These are now explicit
settings, preserved for continuity. See [workflows](workflows.md) for all
optimizer arguments and loss-component logging.

Sources: [GAN](https://arxiv.org/abs/1406.2661),
[R1 regularization](https://arxiv.org/abs/1801.04406).

## What the comparison measures

`study.py` runs explicit `TrainConfig`s through the shared `train.py` loop. Final EMA weights at
the configured budget are evaluated; intermediate sample metrics do not select
a checkpoint. Every run logs the full configuration, objective history, model
size, synchronized training time, generation time, and common sample scores.
`last.pt` also holds optimizers, online weights, EMA, and private RNG states for
resuming the exact same budget. Changing a budget changes the LR schedule and
requires a fresh experiment.

Diffusion and flow can initialize a separate phase with `--init-checkpoint`.
This loads the parent EMA and creates a fresh optimizer; it is different from
`--resume`. Summaries retain `parent_steps`, `total_steps`, and `total_train_s`.
The family, data shape, and diffusion prediction target must match. Resuming
that new phase needs only its own `last.pt`, even if the parent is moved.

Equal update counts and batch sizes are not equal FLOPs. The GAN trains a
second network. Priming's post-training computes frozen features and generated
negative references. The VAE trains an encoder that it does not use to generate.
Report these costs and model sizes instead of calling the comparison compute
matched. The initial MNIST comparison has one training seed: it is exploratory,
not evidence of a statistically significant ranking of model families.

Pixel SWD uses common fixed projections. For ring8, the report additionally
counts points near the eight true component centers so a diffuse ring cannot
hide behind a low projection distance. Image feature distance uses an independent
small dataset-specific classifier. This is **not Inception FID**. Confidence
and class histograms are useful diagnostics, not ground-truth generated-image
accuracy. Always inspect the unfiltered, fixed-seed image grids.
