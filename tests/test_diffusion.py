import pytest
import torch
from torch import nn

from xdiffusion.diffusion import Diffusion, DiffusionConfig


def test_cosine_schedule_has_clean_origin_and_nearly_gaussian_endpoint():
    diffusion = Diffusion(DiffusionConfig(steps=100))
    a = diffusion.alpha_bar
    assert a[0] == 1
    assert 0 < a[-1] < 1e-5
    assert torch.all(a[1:] < a[:-1])
    assert torch.isfinite(a).all()


@pytest.mark.parametrize("shape", [(7, 2), (7, 3, 8, 8)])
def test_forward_process_matches_closed_form_and_clean_boundary(shape):
    diffusion = Diffusion(DiffusionConfig(steps=10))
    clean, noise = torch.randn(shape), torch.randn(shape)
    t = torch.arange(7)
    a = diffusion.alpha_bar[t].reshape(7, *([1] * (len(shape) - 1)))
    noisy = diffusion.add_noise(clean, t, noise)
    torch.testing.assert_close(noisy, a.sqrt() * clean + (1 - a).sqrt() * noise)
    torch.testing.assert_close(noisy[0], clean[0])


class Oracle(nn.Module):
    def __init__(self, diffusion, clean):
        super().__init__()
        self.diffusion, self.clean = diffusion, clean

    def forward(self, noisy, time):
        t = (time * self.diffusion.cfg.steps).round().long()
        a = self.diffusion.alpha_bar[t].reshape(-1, *([1] * (noisy.ndim - 1)))
        return (noisy - a.sqrt() * self.clean) / (1 - a).sqrt()


def test_oracle_has_zero_noise_prediction_loss():
    diffusion = Diffusion(DiffusionConfig(steps=16))
    clean = torch.randn(8, 2)
    loss = diffusion.loss(Oracle(diffusion, clean), clean, torch.arange(1, 9), torch.randn_like(clean))
    assert loss.item() < 1e-10


@pytest.mark.parametrize("sampler,steps", [("ddpm", 16), ("ddim", 16), ("ddim", 4), ("ddim", 1)])
@pytest.mark.parametrize("shape", [(8, 2), (8, 1, 4, 4)])
def test_oracle_reverse_process_recovers_clean_data(sampler, steps, shape):
    diffusion = Diffusion(DiffusionConfig(steps=16))
    clean = torch.full(shape, 0.3)
    model = Oracle(diffusion, clean)
    result = diffusion.sample(model, shape, torch.Generator().manual_seed(42), sampler, steps)
    torch.testing.assert_close(result, clean, atol=2e-4, rtol=0)


def test_ddpm_one_step_matches_posterior_mean_and_variance():
    # Force the first transition to be observable by recording its input at t=T-1.
    diffusion = Diffusion(DiffusionConfig(steps=2))
    seen = []

    class ZeroNoise(nn.Module):
        def forward(self, x, t):
            seen.append(x.clone())
            return torch.zeros_like(x)

    rng = torch.Generator().manual_seed(7)
    initial = torch.randn(32, 2, generator=rng)
    z = torch.randn(32, 2, generator=rng)
    a, previous = diffusion.alpha_bar[2], diffusion.alpha_bar[1]
    alpha, beta = a / previous, 1 - a / previous
    mean = initial / alpha.sqrt()  # DDPM epsilon parameterization, epsilon=0
    variance = beta * (1 - previous) / (1 - a)
    diffusion.sample(ZeroNoise(), (32, 2), torch.Generator().manual_seed(7), "ddpm")
    torch.testing.assert_close(seen[0], initial)
    torch.testing.assert_close(seen[1], mean + variance.sqrt() * z, atol=1e-4, rtol=1e-5)


def test_sampling_uses_only_the_supplied_rng():
    diffusion = Diffusion(DiffusionConfig(steps=8))

    class Zero(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    state = torch.random.get_rng_state()
    a = diffusion.sample(Zero(), (8, 2), torch.Generator().manual_seed(0), steps=4)
    b = diffusion.sample(Zero(), (8, 2), torch.Generator().manual_seed(0), steps=4)
    assert torch.equal(a, b)
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="DDPM requires"):
        diffusion.sample(Zero(), (8, 2), torch.Generator(), sampler="ddpm", steps=4)
    with pytest.raises(ValueError, match="sampling steps"):
        diffusion.sample(Zero(), (8, 2), torch.Generator(), steps=0)


@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
def test_image_sampling_clips_clean_predictions_at_the_last_step(sampler):
    diffusion = Diffusion(DiffusionConfig(steps=8))
    clean = torch.full((4, 1, 4, 4), 2.0)
    model = Oracle(diffusion, clean)
    clipped = diffusion.sample(model, tuple(clean.shape), torch.Generator(), sampler=sampler, clip=True)
    unclipped = diffusion.sample(model, tuple(clean.shape), torch.Generator(), sampler=sampler, clip=False)
    torch.testing.assert_close(clipped, torch.ones_like(clean), atol=1e-5, rtol=0)
    torch.testing.assert_close(unclipped, clean, atol=2e-4, rtol=0)


class VOracle(nn.Module):
    def __init__(self, diffusion, clean):
        super().__init__()
        self.diffusion, self.clean = diffusion, clean

    def forward(self, noisy, time):
        t = (time * self.diffusion.cfg.steps).round().long()
        a = self.diffusion.alpha_bar[t].reshape(-1, *([1] * (noisy.ndim - 1)))
        return (a.sqrt() * noisy - self.clean) / (1 - a).sqrt()


@pytest.mark.parametrize("shape", [(8, 2), (8, 1, 8, 8)])
def test_v_objective_and_clean_noise_conversions_match_closed_forms(shape):
    diffusion = Diffusion(DiffusionConfig(steps=200, prediction="v"))
    clean, noise = torch.randn(shape), torch.randn(shape)
    times = torch.tensor([1, 10, 50, 100, 150, 180, 199, 200])
    noisy = diffusion.add_noise(clean, times, noise)
    oracle = VOracle(diffusion, clean)
    assert diffusion.loss(oracle, clean, times, noise).item() < 1e-9
    torch.testing.assert_close(diffusion.predict_clean(oracle, noisy, times), clean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(diffusion.predict_noise(oracle, noisy, times), noise, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("sampler,steps", [("ddim", 1), ("ddim", 10), ("ddpm", 200)])
def test_v_sampling_recovers_clean_data_even_at_the_almost_zero_snr_endpoint(sampler, steps):
    diffusion = Diffusion(DiffusionConfig(steps=200, prediction="v"))
    clean = torch.full((8, 2), 0.3)
    oracle = VOracle(diffusion, clean)
    result = diffusion.sample(oracle, tuple(clean.shape), torch.Generator(), sampler, steps)
    torch.testing.assert_close(result, clean, atol=1e-5, rtol=1e-5)
