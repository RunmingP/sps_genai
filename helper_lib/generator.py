import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def generate_samples(
    model,
    device: str = "cpu",
    num_samples: int = 10,
    latent_dim: int = 20,
):
    """Sample images from a VAE (.decode) or GAN (.generator) and plot a grid."""
    model.eval()
    model.to(device)

    if hasattr(model, "decode") and callable(getattr(model, "decode")):
        z = torch.randn(num_samples, latent_dim, device=device)
        imgs = model.decode(z).view(-1, 1, 28, 28).detach().cpu()
    elif hasattr(model, "generator") and hasattr(model, "z_dim"):
        z = torch.randn(num_samples, getattr(model, "z_dim", 100), device=device)
        imgs = model.generator(z).view(-1, 1, 28, 28).detach().cpu()
    elif hasattr(model, "generator"):
        z = torch.randn(num_samples, 100, device=device)
        imgs = model.generator(z).view(-1, 1, 28, 28).detach().cpu()
    else:
        raise ValueError("Model must expose .decode(z) or .generator(z).")

    cols = max(1, int(num_samples ** 0.5))
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i, 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return imgs
