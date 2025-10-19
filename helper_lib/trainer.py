# helper_lib/trainer.py
import torch
import torch.nn.functional as F
from typing import Callable, Optional


def train_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 10,
    log_interval: int = 100,
):
    """
    Generic supervised training loop for classification models.
    """
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        total, correct = 0, 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if outputs.ndim == 2 and outputs.size(1) > 1:
                _, preds = outputs.max(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            if (batch_idx + 1) % log_interval == 0:
                if total > 0:
                    acc = 100.0 * correct / total
                    print(
                        f"Epoch {epoch} [{batch_idx+1}/{len(data_loader)}] "
                        f"loss={loss.item():.4f} acc={acc:.2f}%"
                    )
                else:
                    print(
                        f"Epoch {epoch} [{batch_idx+1}/{len(data_loader)}] "
                        f"loss={loss.item():.4f}"
                    )

        epoch_loss = running_loss / len(data_loader.dataset)
        if total > 0:
            epoch_acc = 100.0 * correct / total
            print(f"==> Epoch {epoch} done: loss={epoch_loss:.4f} acc={epoch_acc:.2f}%")
        else:
            print(f"==> Epoch {epoch} done: loss={epoch_loss:.4f}")

    return model


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    VAE loss = reconstruction (BCE) + KL divergence.
    The loss is averaged per batch.
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


def train_vae_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 10,
    log_interval: int = 100,
    criterion: Optional[Callable] = None,
):
    """
    Training loop for a VAE model whose forward returns (recon, mu, logvar).
    If `criterion` is provided, it should accept (recon, x, mu, logvar) and return a scalar loss.
    Otherwise, this uses the default `vae_loss`.
    """
    loss_fn = criterion or vae_loss
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            loss = loss_fn(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch} [{batch_idx+1}/{len(data_loader)}] "
                    f"VAE loss={loss.item():.4f}"
                )

        epoch_loss = running_loss / len(data_loader.dataset)
        print(f"==> Epoch {epoch} done: VAE loss={epoch_loss:.4f}")

    return model
