import torch
from torchvision.datasets import MNIST, EMNIST, KMNIST
from torchvision import transforms
from torchvision.transforms.v2 import GaussianNoise


def get_dataset(
    name="mnist",
    root="./data",
    add_noise=False,
    sigma=None,
    seed=None,
    perturb=False,
):
    """
    Returns the specified dataset with optional Gaussian noise added.
    Args:
        name (str): Name of the dataset ('mnist', 'emnist', 'kmnist').
        root (str): Root directory for dataset storage.
        add_noise (bool): Whether to add Gaussian noise to the images.
        sigma (float): Standard deviation of the Gaussian noise.
        seed (int): Random seed for reproducibility.
        perturb (bool): Whether to add a small random perturbation to the images.

    NOTE: may require clipping after adding noise to keep pixel values valid.
    """
    if seed is not None:
        torch.manual_seed(seed)

    transform_list = []
    if add_noise and sigma is not None:
        transform_list.append(GaussianNoise(0.0, sigma))
    transform_list.append(transforms.ToTensor())
    if perturb:
        transform_list.append(
            transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))
        )
    transform = transforms.Compose(transform_list)

    if name == "mnist":
        dataset = MNIST(root=root, train=True, download=True, transform=transform)
    elif name == "emnist":
        dataset = EMNIST(
            root=root, split="letters", train=True, download=True, transform=transform
        )
    elif name == "kmnist":
        dataset = KMNIST(root=root, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {name} is not supported.")
    return dataset


def get_dataloader(
    name="mnist",
    root="./data",
    add_noise=False,
    sigma=None,
    batch_size=32,
    seed=None,
    shuffle=True,
    perturb=False,
):
    """
    Returns a DataLoader for the specified dataset.
    Args:
        name (str): Name of the dataset ('mnist', 'emnist', 'kmnist').
        root (str): Root directory for dataset storage.
        add_noise (bool): Whether to add Gaussian noise to the images.
        sigma (float): Standard deviation of the Gaussian noise.
        batch_size (int): Number of samples per batch.
        seed (int): Random seed for reproducibility.
        shuffle (bool): Whether to shuffle the dataset.
        perturb (bool): Whether to add a small random perturbation to the images.
    """
    dataset = get_dataset(
        name=name,
        root=root,
        add_noise=add_noise,
        sigma=sigma,
        seed=seed,
        perturb=perturb,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return dataloader
