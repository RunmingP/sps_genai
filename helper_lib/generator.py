import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def generate_samples(model, device: str, num_samples: int = 10, diffusion_steps: int = 100):
    model.to(device)
    model.eval()

    betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    x = torch.randn(num_samples, 1, 28, 28, device=device)

    for t in reversed(range(diffusion_steps)):
        t_index = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps = model(x, t_index)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps
        )

        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)

    imgs = x.detach().cpu()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)

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


def generate_ebm_samples(
    model,
    device: str = "cpu",
    num_samples: int = 10,
    steps: int = 50,
    step_size: float = 0.1,
    init_std: float = 1.0,
    add_noise: bool = False,
    noise_std: float = 0.01,
):
    """
    Energy-Based Model sampling via gradient descent on the input.
    Returns: imgs of shape (num_samples, 1, 28, 28) normalized to [0, 1].
    """
    model.to(device)
    model.eval()

    x = torch.randn(num_samples, 1, 28, 28, device=device) * init_std
    x = x.clone().detach().requires_grad_(True)

    for _ in range(steps):
        energy = model(x).sum()
        energy.backward()

        with torch.no_grad():
            x = x - step_size * x.grad
            if add_noise:
                x = x + noise_std * torch.randn_like(x)
            x = torch.clamp(x, -3.0, 3.0)

        x.grad.zero_()

    imgs = x.detach().cpu()
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
    return imgs
