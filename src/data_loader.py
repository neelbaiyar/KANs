import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size=128, small=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if small:
        train_ds, _ = random_split(train_ds, [10000, len(train_ds) - 10000])
        test_ds,  _ = random_split(test_ds,  [2000, len(test_ds) - 2000])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = 28*28
    num_classes = 10
    return train_loader, test_loader, input_dim, num_classes

def get_cifar10_loaders(batch_size=128, small=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    if small:
        train_ds, _ = random_split(train_ds, [10000, len(train_ds) - 10000])
        test_ds, _  = random_split(test_ds,  [2000,  len(test_ds)  - 2000])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    input_dim = 32 * 32 * 3   
    num_classes = 10

    return train_loader, test_loader, input_dim, num_classes
