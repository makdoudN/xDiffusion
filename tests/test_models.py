import pytest
import torch
from jaxtyping import TypeCheckError

from xdiffusion.diffusion import Diffusion, DiffusionConfig
from xdiffusion.models import MLP, ModelConfig, make_model


@pytest.mark.parametrize("shape", [(2,), (1, 16, 16), (1, 28, 28), (3, 32, 32)])
def test_denoiser_preserves_shape_and_all_parameters_receive_gradients(shape):
    model = make_model(shape, ModelConfig(width=16))
    x = torch.randn(3, *shape)
    result = model(x, torch.rand(3))
    assert result.shape == x.shape
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mlp_learns_a_fixed_denoising_problem():
    torch.manual_seed(0)
    model = MLP(ModelConfig(width=32, depth=2))
    diffusion = Diffusion(DiffusionConfig(steps=16))
    clean, noise = torch.zeros(64, 2), torch.randn(64, 2)
    t = torch.full((64,), 8, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial = diffusion.loss(model, clean, t, noise).item()
    for _ in range(60):
        optimizer.zero_grad()
        diffusion.loss(model, clean, t, noise).backward()
        optimizer.step()
    assert diffusion.loss(model, clean, t, noise).item() < initial * 0.05


@pytest.mark.parametrize("kwargs", [{"width": 7}, {"depth": 0}, {"time_dim": 3}])
def test_invalid_architecture_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_pytest_hook_checks_tensor_dtypes_and_batch_axes():
    model = MLP(ModelConfig(width=16))
    with pytest.raises(TypeCheckError):
        model(torch.zeros(3, 2, dtype=torch.long), torch.rand(3))
    with pytest.raises(TypeCheckError):
        model(torch.randn(3, 2), torch.rand(4))


def test_mlp_preserves_the_high_noise_identity_boundary():
    model = MLP(ModelConfig(width=16))
    noise = torch.randn(8, 2)
    torch.testing.assert_close(model(noise, torch.ones(8)), noise)
