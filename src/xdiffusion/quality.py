"""Small held-out image judge, distinct from the encoder used to train Priming.

The feature distance here is dataset-specific, not Inception FID. Classifier
confidence is a diagnostic, not a certificate that a generated image is good.
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import tyro
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from xdiffusion.common import get_device, write_json
from xdiffusion.data import DataConfig, Split, load_data


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.embedding = nn.Linear(64 * 7 * 7, 64)
        self.head = nn.Linear(64, 10)

    def features(self, x: Float[Tensor, " batch 1 28 28"]) -> Float[Tensor, " batch 64"]:
        return self.representations(x)[-1]

    def representations(self, x: Float[Tensor, " batch 1 28 28"]) -> tuple[Tensor, Tensor, Tensor]:
        early = self.conv[:3](x)
        late = self.conv[3:](early)
        embedding = F.silu(self.embedding(rearrange(late, "b c h w -> b (c h w)")))
        return (
            F.adaptive_avg_pool2d(early, 4).flatten(1),
            F.adaptive_avg_pool2d(late, 4).flatten(1),
            embedding,
        )

    def forward(self, x: Float[Tensor, " batch 1 28 28"]) -> Float[Tensor, " batch 10"]:
        return self.head(self.features(x))


def load_labels(cfg: DataConfig, split: Split) -> Int[Tensor, " sample"]:
    from torchvision.datasets import MNIST, FashionMNIST

    if cfg.task not in ("mnist", "fashion"):
        raise ValueError("the small image judge supports MNIST and Fashion-MNIST")
    cls = MNIST if cfg.task == "mnist" else FashionMNIST
    labels = cls(str(cfg.root), train=split != "test", download=False).targets
    if split != "test":
        order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(cfg.seed))
        labels = labels[order[5000:] if split == "train" else order[:5000]]
    return labels.long()


@torch.no_grad()
def accuracy(
    model: nn.Module, data: Float[Tensor, " sample 1 28 28"], labels: Int[Tensor, " sample"]
) -> float:
    device = next(model.parameters()).device
    correct = 0
    for x, y in zip(data.split(256), labels.split(256), strict=True):
        correct += (model(x.to(device)).argmax(-1).cpu() == y).sum().item()
    return correct / len(data)


def load_encoder(path: Path, device: torch.device, task: str | None = None) -> FeatureNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if task is not None and checkpoint["task"] != task:
        raise ValueError(f"encoder is for {checkpoint['task']}, requested {task}")
    model = FeatureNet().to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False)


def frechet_distance(x: Float[Tensor, " sample features"], y: Float[Tensor, " other features"]) -> float:
    """PSD square roots via symmetric eigendecompositions, in CPU float64."""
    x, y = x.cpu().double(), y.cpu().double()
    cx, cy = torch.cov(x.T), torch.cov(y.T)
    values, vectors = torch.linalg.eigh(cx)
    root = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    middle = root @ cy @ root
    trace_root = torch.linalg.eigvalsh((middle + middle.T) / 2).clamp_min(0).sqrt().sum()
    distance = (x.mean(0) - y.mean(0)).square().sum() + cx.trace() + cy.trace() - 2 * trace_root
    return distance.clamp_min(0).item()


@torch.no_grad()
def image_scores(
    judge: FeatureNet,
    real: Float[Tensor, " sample 1 28 28"],
    generated: Float[Tensor, " sample 1 28 28"],
) -> dict:
    device = next(judge.parameters()).device
    features = []
    for images in (real, generated):
        features.append(torch.cat([judge.features(b.to(device)).cpu() for b in images.split(256)]))
    real_features, fake_features = features
    probabilities = judge.head(fake_features.to(device)).softmax(-1).cpu()
    counts = probabilities.argmax(-1).bincount(minlength=10)
    distribution = counts.float() / len(generated)
    return {
        "feature_fd": frechet_distance(real_features, fake_features),
        "confident_fraction": (probabilities.max(-1).values >= 0.95).float().mean().item(),
        "class_counts": counts.tolist(),
        "class_entropy": -(distribution * distribution.clamp_min(1e-12).log()).sum().item(),
    }


@dataclass
class QualityConfig:
    data: DataConfig = field(default_factory=lambda: DataConfig(task="mnist"))
    out_dir: Path | None = None  # default: out/encoders/<task>
    steps: int = 2500
    batch_size: int = 128
    device: str = "auto"
    threads: int = 4


def main(cfg: QualityConfig) -> None:
    cfg = replace(cfg, out_dir=cfg.out_dir or Path("out/encoders") / cfg.data.task)
    torch.set_num_threads(cfg.threads)
    if cfg.steps < 1 or cfg.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if (cfg.out_dir / "judge.pt").exists():
        raise FileExistsError(f"encoders already exist: {cfg.out_dir}")
    device = get_device(cfg.device)
    train, val = load_data(cfg.data), load_data(cfg.data, "val")
    labels, val_labels = load_labels(cfg.data, "train"), load_labels(cfg.data, "val")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    # Independent weights and random streams. Only `priming.pt` enters generator training.
    for name, seed in (("judge", 42), ("priming", 314)):
        torch.manual_seed(seed)
        model = FeatureNet().to(device)
        rng = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(1, cfg.steps + 1):
            indices = torch.randint(len(train), (cfg.batch_size,), generator=rng)
            logits = model(train[indices].to(device))
            loss = F.cross_entropy(logits, labels[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                print(f"{name} step {step} | CE {loss.item():.4f}", flush=True)
        model.eval()
        score = accuracy(model, val, val_labels)
        torch.save(
            {"model": model.state_dict(), "task": cfg.data.task, "seed": seed}, cfg.out_dir / f"{name}.pt"
        )
        results[name] = {"seed": seed, "val_accuracy": score}
        print(f"{name} validation accuracy {score:.4%}", flush=True)
    write_json(cfg.out_dir / "config.json", cfg)
    write_json(cfg.out_dir / "summary.json", {"data": asdict(cfg.data), "encoders": results})


if __name__ == "__main__":
    main(tyro.cli(QualityConfig))
