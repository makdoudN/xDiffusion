import pytest
import torch
from jaxtyping import TypeCheckError

from xdiffusion.diffusion import Diffusion, DiffusionConfig
from xdiffusion.metrics import noise_mse, sliced_wasserstein
from xdiffusion.models import MLP, ModelConfig


def test_swd_is_zero_for_permuted_identical_data_and_positive_for_translation():
    x = torch.randn(128, 2)
    assert sliced_wasserstein(x, x.flip(0)) == 0
    assert sliced_wasserstein(x, x + 2) > 1
    assert sliced_wasserstein(x, x + 2) == sliced_wasserstein(x + 2, x)
    with pytest.raises(TypeCheckError):
        sliced_wasserstein(x, x[:2])


def test_validation_is_repeatable_preserves_training_rng_and_counts_partial_batch():
    model = MLP(ModelConfig(width=16)).eval()
    diffusion = Diffusion(DiffusionConfig(steps=10))
    data = torch.randn(19, 2)
    state = torch.random.get_rng_state()
    result = noise_mse(model, diffusion, data, batch_size=8)
    assert result == noise_mse(model, diffusion, data, batch_size=8)
    assert torch.equal(state, torch.random.get_rng_state())
    # Reconstruct the independently generated batches, including the final 3 samples.
    rng = torch.Generator().manual_seed(0)
    losses = []
    for batch in data.split(8):
        t = torch.randint(1, 11, (len(batch),), generator=rng)
        noise = torch.randn(batch.shape, generator=rng)
        losses.append(diffusion.loss(model, batch, t, noise).item() * len(batch))
    assert result == pytest.approx(sum(losses) / 19)
