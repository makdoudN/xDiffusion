"""Non-saturating logistic GAN; a learned critic, not a reconstruction objective."""

from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, shape: tuple[int, ...], width: int = 32):
        super().__init__()
        if len(shape) == 1:
            self.net = nn.Sequential(
                nn.Linear(shape[0], width),
                nn.LeakyReLU(0.2),
                nn.Linear(width, width),
                nn.LeakyReLU(0.2),
                nn.Linear(width, 1),
            )
        else:
            channels, height, side = shape
            self.net = nn.Sequential(
                nn.Conv2d(channels, width, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(2 * width * (height // 4) * (side // 4), 1),
            )

    def forward(self, x: Float[Tensor, " batch *shape"]) -> Float[Tensor, " batch"]:
        return self.net(x).squeeze(-1)


def discriminator_loss(
    real_logits: Float[Tensor, " batch"], fake_logits: Float[Tensor, " batch"]
) -> Float[Tensor, ""]:
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def generator_loss(fake_logits: Float[Tensor, " batch"]) -> Float[Tensor, ""]:
    return F.softplus(-fake_logits).mean()
