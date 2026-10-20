"""A convolutional VAE for images and a small MLP VAE for coordinates."""

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class Decoder(nn.Module):
    """Shared decoder topology for the VAE and GAN, without shared weights."""

    def __init__(self, shape: tuple[int, ...], width: int, latent_dim: int):
        super().__init__()
        self.shape = shape
        self.width = width
        if len(shape) == 1:
            self.project = nn.Sequential(nn.Linear(latent_dim, width), nn.SiLU(), nn.Linear(width, width))
            self.output = nn.Sequential(nn.SiLU(), nn.Linear(width, shape[0]))
        else:
            channels, height, side = shape
            if height % 4 or side % 4:
                raise ValueError("image dimensions must be divisible by four")
            self.project = nn.Linear(latent_dim, 2 * width * (height // 4) * (side // 4))
            self.output = nn.Sequential(
                nn.GroupNorm(8, 2 * width),
                nn.SiLU(),
                nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(),
                nn.ConvTranspose2d(width, channels, 4, stride=2, padding=1),
                nn.Tanh(),
            )

    def forward(self, z: Float[Tensor, " batch latent"]) -> Float[Tensor, " batch *shape"]:
        h = self.project(z)
        if len(self.shape) == 3:
            h = rearrange(
                h, "b (c h w) -> b c h w", c=2 * self.width, h=self.shape[1] // 4, w=self.shape[2] // 4
            )
        return self.output(h)


class VAE(nn.Module):
    def __init__(self, shape: tuple[int, ...], width: int = 32, latent_dim: int = 32):
        super().__init__()
        self.shape, self.latent_dim = shape, latent_dim
        if len(shape) == 1:
            self.encoder = nn.Sequential(
                nn.Linear(shape[0], width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
            )
            features = width
        else:
            channels, height, side = shape
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, width, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.Flatten(),
            )
            features = 2 * width * (height // 4) * (side // 4)
        self.mean = nn.Linear(features, latent_dim)
        self.log_variance = nn.Linear(features, latent_dim)
        self.decoder = Decoder(shape, width, latent_dim)

    def encode(
        self, x: Float[Tensor, " batch *shape"]
    ) -> tuple[Float[Tensor, " batch latent"], Float[Tensor, " batch latent"]]:
        h = self.encoder(x)
        return self.mean(h), self.log_variance(h).clamp(-20, 10)

    def forward(
        self, x: Float[Tensor, " batch *shape"], noise: Float[Tensor, " batch latent"]
    ) -> tuple[
        Float[Tensor, " batch *shape"], Float[Tensor, " batch latent"], Float[Tensor, " batch latent"]
    ]:
        mean, log_variance = self.encode(x)
        z = mean + (0.5 * log_variance).exp() * noise
        return self.decoder(z), mean, log_variance

    def reconstruct(self, x: Float[Tensor, " batch *shape"]) -> Float[Tensor, " batch *shape"]:
        return self.decoder(self.encode(x)[0])


def vae_loss(
    model: VAE,
    clean: Float[Tensor, " batch *shape"],
    noise: Float[Tensor, " batch latent"],
    beta: float = 1.0,
) -> Float[Tensor, ""]:
    output, mean, log_variance = model(clean, noise)
    if clean.ndim == 4:
        # Bernoulli pixel likelihood proxy: sum features, then average examples.
        reconstruction = F.binary_cross_entropy(
            ((output + 1) / 2).clamp(1e-6, 1 - 1e-6), (clean + 1) / 2, reduction="sum"
        ) / len(clean)
    else:
        # A fixed Gaussian decoder with standard deviation 0.1 in coordinate units.
        reconstruction = (output - clean).square().sum(-1).mean() / (2 * 0.1**2)
    kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(-1).mean()
    return reconstruction + beta * kl
