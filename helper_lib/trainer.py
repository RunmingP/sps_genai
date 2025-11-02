import torch
import torch.nn.functional as F
import torch.nn as nn
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


def train_gan(
    model,
    data_loader: torch.utils.data.DataLoader,
    criterion: Optional[Callable] = None,
    optimizer=None,
    device: str = "cpu",
    epochs: int = 10,
    z_dim: int = 100,
    lr: float = 2e-4,
    betas=(0.5, 0.999),
    log_interval: int = 100,
):
    model.to(device)
    model.train()
    G, D = model.generator, model.discriminator
    loss_fn = criterion or nn.BCEWithLogitsLoss()

    if optimizer is None:
        optD = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
        optG = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    elif isinstance(optimizer, dict) and "D" in optimizer and "G" in optimizer:
        optD, optG = optimizer["D"], optimizer["G"]
    else:
        raise ValueError("optimizer must be None or a dict with keys 'D' and 'G'.")

    for ep in range(1, epochs + 1):
        D_running, G_running, seen = 0.0, 0.0, 0

        for batch_idx, (real, _) in enumerate(data_loader):
            real = real.to(device)
            bs = real.size(0)
            seen += bs

            z = torch.randn(bs, z_dim, device=device)
            with torch.no_grad():
                fake = G(z)

            real_logits = D(real)
            fake_logits = D(fake)

            d_real = loss_fn(real_logits, torch.ones_like(real_logits))
            d_fake = loss_fn(fake_logits, torch.zeros_like(fake_logits))
            d_loss = d_real + d_fake

            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()

            z = torch.randn(bs, z_dim, device=device)
            fake = G(z)
            g_logits = D(fake)
            g_loss = loss_fn(g_logits, torch.ones_like(g_logits))

            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()

            D_running += d_loss.item() * bs
            G_running += g_loss.item() * bs

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Epoch {ep} [{batch_idx+1}/{len(data_loader)}] "
                    f"D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}"
                )

        d_avg = D_running / max(1, seen)
        g_avg = G_running / max(1, seen)
        print(f"==> Epoch {ep} done: D_loss={d_avg:.4f} G_loss={g_avg:.4f}")

    return model
