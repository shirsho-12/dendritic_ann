import torch
from datasets import get_dataloader
from model import DANN
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        learning_rate=0.001,
        dataset_name="mnist",
        data_root="./data",
        add_noise=False,
        sigma=None,
        batch_size=32,
        weight_decay=0.0001,
        seed=None,
        shuffle=True,
        perturb=False,
    ):
        self.dataloader: DataLoader = get_dataloader(
            name=dataset_name,
            root=data_root,
            add_noise=add_noise,
            sigma=sigma,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            perturb=perturb,
        )
        self.model: nn.Module = DANN(
            input_dim=(28, 28),
            dends=[128, 64],
            soma=[64, 32],
            num_classes=10,
            seed=seed,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, num_epochs: int):
        self.model.train()
        for epoch in range(1, num_epochs + 1):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{num_epochs}")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch} completed. Loss: {loss.item():.4f}")

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
