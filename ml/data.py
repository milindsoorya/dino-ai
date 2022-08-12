import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split


def load_mnist_data(root='data', flatten=True, batch_size=32):
    if flatten:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
        )
    else:
        transform = torchvision.transforms.ToTensor(),

    train_dataset = MNIST(root=root, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False,
                         download=True, transform=transform)

    # Create validation split
    m = len(train_dataset)

    train_data, val_data = random_split(
        train_dataset, [int(m-m*0.2), int(m*0.2)])

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    valid_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader
