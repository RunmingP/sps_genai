import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class ResNet18_1ch(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h3))

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = recon.view(-1, 1, 28, 28)
        return recon, mu, logvar


class CNN64x64_RGB(nn.Module):
    """
    Assignment CNN:
    Input: 64x64x3
    Conv2D(16, 3x3, s=1, p=1) -> ReLU -> MaxPool(2x2, s=2)
    Conv2D(32, 3x3, s=1, p=1) -> ReLU -> MaxPool(2x2, s=2)
    Flatten -> FC(100) -> ReLU -> FC(10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class GANGenerator(nn.Module):
    """
    DCGAN Generator for MNIST (1x28x28)
    z (B,100) -> FC to 7*7*128 -> reshape -> ConvT(128->64,k4 s2 p1) + BN + ReLU
              -> ConvT(64->1,k4 s2 p1) + Tanh
    Output range: [-1, 1]
    """
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 128),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.deconv(x)
        return x  


class GANDiscriminator(nn.Module):
    """
    DCGAN Discriminator for MNIST (1x28x28)
    1x28x28 -> Conv(1->64,k4 s2 p1) + LeakyReLU(0.2)
            -> Conv(64->128,k4 s2 p1) + BN + LeakyReLU(0.2)
            -> Flatten -> Linear(128*7*7->1)  # logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),  
        )

    def forward(self, x):
        return self.net(x)


class GAN(nn.Module):
    """
    Wrapper for DCGAN parts, keeping the same interface as before.
    forward(z) -> (fake_imgs, d_logits_on_fake)
    """
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = GANGenerator(z_dim)
        self.discriminator = GANDiscriminator()

    def forward(self, z):
        fake = self.generator(z)
        d_out = self.discriminator(fake)
        return fake, d_out


def get_model(model_name: str):
    name = model_name.strip().lower()
    if name in ("cnn", "cnn64", "cnn64x64", "cnn64x64_rgb"):
        return CNN64x64_RGB(num_classes=10)
    elif name == "mnist_cnn":
        return CNN()
    elif name == "enhancedcnn":
        return EnhancedCNN()
    elif name == "resnet18":
        return ResNet18_1ch(num_classes=10)
    elif name == "vae":
        return VAE(latent_dim=20)
    elif name == "gan":
        return GAN(z_dim=100)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
