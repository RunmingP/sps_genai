import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True):
    """
    MNIST loader kept for earlier modules.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        transform=transform,
        download=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    data_dir: str,
    batch_size: int = 128,
    img_size: int = 64,
    augment: bool = True,
):
    """
    Return (train_loader, test_loader) for CIFAR-10, resizing to 64x64.
    """
    train_tf = []
    if augment:
        train_tf += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
    # resize to 64x64 as required by the assignment
    train_tf += [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    train_tf = transforms.Compose(train_tf)

    train_set = datasets.CIFAR10(root=data_dir, train=True,  transform=train_tf, download=True)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, transform=test_tf,  download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, test_loader


def cifar10_classes():
    return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

