import pytest
import torch
from torch import nn

from xdiffusion.flow import FlowConfig, flow_loss, integrate
from xdiffusion.gan import discriminator_loss, generator_loss
from xdiffusion.priming import FrozenFeatures, laplacian_mmd, reconstruction_loss
from xdiffusion.quality import FeatureNet, frechet_distance, load_encoder
from xdiffusion.vae import VAE, vae_loss


class ConstantVelocity(nn.Module):
    def forward(self, x, t):
        return torch.ones_like(x) * 2


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_flow_integrates_known_field_and_reverses_it(solver):
    initial = torch.randn(8, 2)
    cfg = FlowConfig(steps=8, solver=solver)
    generated = integrate(ConstantVelocity(), initial, cfg)
    torch.testing.assert_close(generated, initial + 2)
    torch.testing.assert_close(integrate(ConstantVelocity(), generated, cfg, 1.0, 0.0), initial)


def test_flow_matching_uses_the_correct_path_and_velocity():
    clean, noise = torch.full((4, 2), 3.0), torch.ones(4, 2)
    seen = []

    class Oracle(nn.Module):
        def forward(self, x, t):
            seen.append(x)
            return clean - noise

    times = torch.tensor([0.0, 0.25, 0.75, 1.0])
    assert flow_loss(Oracle(), clean, noise, times).item() == 0
    torch.testing.assert_close(seen[0], (1 - times[:, None]) * noise + times[:, None] * clean)


def test_heun_is_more_accurate_than_euler_for_a_linear_field():
    class Linear(nn.Module):
        def forward(self, x, t):
            return x

    x = torch.ones(4, 2)
    target = x * torch.exp(torch.tensor(1.0))
    euler = integrate(Linear(), x, FlowConfig(8, "euler"))
    heun = integrate(Linear(), x, FlowConfig(8, "heun"))
    assert (heun - target).abs().max() < (euler - target).abs().max() / 10


def test_priming_reconstruction_uses_the_papers_fixed_noise_mixture():
    clean, noise = torch.randn(4, 2), torch.randn(4, 2)
    seen = []

    class Oracle(nn.Module):
        def forward(self, x):
            seen.append(x)
            return clean

    assert reconstruction_loss(Oracle(), clean, noise).item() == 0
    torch.testing.assert_close(seen[0], 0.2 * clean + 0.8 * noise)


def test_priming_surrogate_has_half_mmd_gradient_and_detaches_reference():
    torch.manual_seed(0)
    online, real = torch.randn(8, 3, requires_grad=True), torch.randn(9, 3, requires_grad=True)
    negative = online.detach().clone().requires_grad_()
    loss = laplacian_mmd(online, real, negative)
    derivative = torch.autograd.grad(loss, (online, real, negative), allow_unused=True)
    assert derivative[1] is None and derivative[2] is None

    def kernel(x, y):
        return torch.exp(-torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist") / 3**0.5)

    full = kernel(online, online).mean() + kernel(real, real).mean() - 2 * kernel(online, real).mean()
    expected = torch.autograd.grad(full / 2, online)[0]
    torch.testing.assert_close(derivative[0], expected)


def test_frozen_priming_encoder_still_backpropagates_to_the_generator_output():
    encoder = FrozenFeatures(FeatureNet(), torch.randn(8, 1, 28, 28))
    fake = torch.randn(4, 1, 28, 28, requires_grad=True)
    encoder(fake).square().mean().backward()
    assert fake.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())


@pytest.mark.parametrize("shape", [(2,), (1, 16, 16), (1, 28, 28), (3, 32, 32)])
def test_vae_trains_reconstructs_and_generates_with_a_real_latent_bottleneck(shape):
    model = VAE(shape, width=8, latent_dim=4)
    clean = torch.rand(3, *shape) * 2 - 1
    loss = vae_loss(model, clean, torch.randn(3, 4))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.reconstruct(clean).shape == clean.shape
    assert model.decoder(torch.randn(3, 4)).shape == clean.shape


def test_gan_logistic_losses_are_numerically_stable_at_extreme_logits():
    logits = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    loss = discriminator_loss(logits, -logits) + generator_loss(logits)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_frechet_distance_is_zero_for_identical_features_and_detects_translation():
    features = torch.randn(64, 4)
    assert frechet_distance(features, features) < 1e-6
    assert frechet_distance(features, features + 2) == pytest.approx(16, abs=1e-5)


def test_vae_elbo_sums_pixels_and_latents_before_averaging_examples():
    model = VAE((1, 16, 16), width=8, latent_dim=2)
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    nn.init.ones_(model.mean.bias)
    # Decoder p(pixel)=0.5 gives 256*log(2); KL(N(ones, I), N(0, I))=1.
    loss = vae_loss(model, torch.zeros(3, 1, 16, 16), torch.zeros(3, 2))
    assert loss.item() == pytest.approx(256 * torch.log(torch.tensor(2.0)).item() + 1, abs=1e-4)


def test_encoder_rejects_a_different_task_or_training_split(tmp_path):
    path = tmp_path / "judge.pt"
    torch.save({"model": FeatureNet().state_dict(), "task": "mnist", "data_seed": 1234}, path)
    load_encoder(path, torch.device("cpu"), "mnist", 1234)
    with pytest.raises(ValueError, match="encoder is for"):
        load_encoder(path, torch.device("cpu"), "fashion", 1234)
    with pytest.raises(ValueError, match="training split differs"):
        load_encoder(path, torch.device("cpu"), "mnist", 42)
