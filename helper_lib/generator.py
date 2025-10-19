import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def generate_samples(
    model,
    device: str = "cpu",
    num_samples: int = 10,
    latent_dim: int = 20,
):
    """
    Sample images from a VAE by drawing z ~ N(0, I) and decoding them.
    Assumes `model.decode(z)` returns images in [0,1] with shape (N, 1, 28, 28).

    Args:
        model: A trained VAE with a `.decode(z)` method.
        device: Torch device to run sampling on.
        num_samples: Number of images to sample and display.
        latent_dim: Dimensionality of the latent vector z.

    Returns:
        A CPU tensor of shape (num_samples, 1, 28, 28) containing sampled images.
    """
    model.eval()
    model.to(device)

    z = torch.randn(num_samples, latent_dim, device=device)
    imgs = model.decode(z).view(-1, 1, 28, 28).detach().cpu()

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
