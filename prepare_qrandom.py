import os

import lightning as L
import torch
import torchvision.transforms as T
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision import datasets


class MnistModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1024,
        root: str = os.path.join("data", "mnist"),
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self._root = root
        os.makedirs(self._root, exist_ok=True)
        transforms = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        self._train_data = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms,
        )
        self._eval_data = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transforms,
        )
        self._test_data = self._eval_data
        self._batch_size = batch_size
        self._num_workers = num_workers

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self._train_data,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self._eval_data,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self._eval_data,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )


@torch.no_grad
def percent_correct(logits: torch.Tensor, targets: torch.Tensor) -> float:
    assert logits.size(0) == targets.size(
        0
    ), f"Expected logits and targets to have the same batch size, but got {logits.size(0)} and {targets.size(0)}"
    correct = torch.mean((torch.argmax(logits, dim=1) == targets).float()).item()
    return correct * 100
