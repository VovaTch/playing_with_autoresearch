import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import lightning as L
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    OptimizerLRScheduler,
)

from prepare_qrandom import Cifar100Module, MnistModule, percent_correct


@dataclass
class NetConfig:
    num_layers: int = 7
    hidden_dim: int = 256
    use_activations: bool = True


@dataclass
class LearningConfig:
    learning_rate: float = 9.0e-2
    epochs: int = 50
    batch_size: int = 1024
    flip_factor: float = 0.1
    min_prob: float = 1e-3
    num_workers: int = 11


def map_to_closest(tensor: torch.Tensor, values: list[float]) -> torch.Tensor:
    """
    Maps the values in a PyTorch tensor to the closest value in a given list.

    Args:
        tensor (torch.Tensor): The input tensor.
        values (list): A list of possible values.

    Returns:
        torch.Tensor: A tensor with values mapped to the closest value in the given list.
    """
    values_tensor = torch.tensor(values).to(tensor.device)
    diffs = torch.abs(tensor.unsqueeze(-1) - values_tensor)
    closest_indices = diffs.argmin(dim=-1)
    mapped_tensor = values_tensor[closest_indices]

    return mapped_tensor


def top_k_samples(value_tensor: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Selects the top k samples from a given value tensor.

    Args:
        value_tensor (torch.Tensor): The input tensor containing the values.
        top_k (int): The number of top samples to select.

    Returns:
        torch.Tensor: A tensor containing the top k samples from the input tensor.
    """
    init_shape = value_tensor.shape
    value_tensor = value_tensor.flatten()
    _, indices = torch.topk(value_tensor, k=top_k)
    mask = get_mask(indices.to(value_tensor.device), value_tensor.numel())
    return (value_tensor * mask).reshape(init_shape)


def get_mask(indices: torch.Tensor, total_params: int) -> torch.Tensor:
    """
    Creates a mask tensor based on the given indices and total number of parameters.

    Args:
        indices (torch.Tensor): A tensor containing the indices of the parameters to be masked.
        total_params (int): The total number of parameters.

    Returns:
        torch.Tensor: A boolean mask tensor with the same shape as `total_params`,
        where the indices specified by `indices` are set to True and the rest are set to False.
    """

    mask = torch.zeros(total_params, dtype=torch.bool).to(indices.device)
    mask.scatter_(0, indices, True)

    return mask


class BinaryRandom(Optimizer):
    """
    Dotan's Binary Random training setup, now as an optimizer that allows it to be used in any arbitrary network.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        max_value: int,
        step_size: int,
        lr=required,
        min_prob: float = 1e-4,
        flip_factor: float = 0.1,
    ) -> None:
        defaults = dict(
            lr=lr,
            max_value=max_value,
            step_size=step_size,
            flip_factor=flip_factor,
            min_prob=min_prob,
        )
        super().__init__(params, defaults)
        value_list = torch.arange(-max_value, max_value + 1, step_size).tolist()

        # Convert all parameters into the correct values
        for group in self.param_groups:
            for p in group["params"]:
                p.data = map_to_closest(
                    torch.randn_like(p.data) * 0.7, value_list
                ).float()

    def __setstate__(self, state: dict[str, Any]) -> None:
        return super().__setstate__(state)

    def step(self, closure: Callable[[], float] | None = None) -> None | float:  # type: ignore
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                state = self.state[p]
                if "momentum_buf" not in state:
                    state["momentum_buf"] = torch.zeros_like(d_p)
                state["momentum_buf"].mul_(0.97).add_(d_p, alpha=0.03)
                d_p = state["momentum_buf"]

                top_k_p = max(1, int(group["lr"] * p.data.numel()))

                weights_sign = torch.sign(p.data)
                update_direction = torch.sign(d_p)
                w_coords_2d = -d_p * weights_sign + (weights_sign == 0) * torch.abs(d_p)
                w_coords_2d = top_k_samples(w_coords_2d, top_k=top_k_p)
                w_coords_2d = self._scale_flip_factor(w_coords_2d, group["min_prob"])
                rand_matrix_W = torch.rand(p.data.shape).to(w_coords_2d.device)
                idx_to_flip_W = rand_matrix_W < w_coords_2d
                weight_update = torch.zeros_like(update_direction)
                weight_update[idx_to_flip_W] -= (
                    update_direction[idx_to_flip_W] * group["step_size"]
                )
                p.data.add_(weight_update)
                p.data = torch.clamp(p.data, -group["max_value"], group["max_value"])

        return loss

    @staticmethod
    def _scale_flip_factor(
        w_coords_2d: torch.Tensor, min_prob: float = 0.0
    ) -> torch.Tensor:
        try:
            max_w_coords = torch.max(w_coords_2d)
            min_positive_w_coords = torch.min(w_coords_2d[w_coords_2d > 0])
            w_coords_2d[w_coords_2d > 0] = (
                (w_coords_2d[w_coords_2d > 0] - min_positive_w_coords)
                / (max_w_coords - min_positive_w_coords)
                * 1.0
            )
            w_coords_2d[w_coords_2d <= 0] = min_prob
        except Exception as e:
            warnings.warn(f"Possible matrix of zeroes, {e}")
        return w_coords_2d


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),  # 32
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU(0.4),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU(0.4),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU(0.4),
            nn.MaxPool2d(2, 2),  # 16
            nn.Conv2d(48, 128, 3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.4),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.4),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.4),
            nn.MaxPool2d(2, 2),  # 8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.4),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, affine=False),
            nn.LeakyReLU(0.4),
            nn.Conv2d(128, 160, 3, padding=1),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(0.4),
            nn.MaxPool2d(2, 2),  # 4
            nn.Conv2d(160, 160, 3, padding=1),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(0.4),
            nn.Flatten(),
            nn.Linear(160 * 4 * 4, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.LeakyReLU(0.4),
            nn.Linear(256, 100),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class FullyConnected(nn.Module):
    def __init__(self, net_cfg: NetConfig) -> None:
        super().__init__()
        self._net_cfg = net_cfg
        layers = []

        if net_cfg.num_layers < 2:
            raise ValueError(
                f"The network must have at least two layers, got {net_cfg.num_layers}"
            )

        layers.append(nn.Linear(784, self._net_cfg.hidden_dim))
        if net_cfg.use_activations:
            layers.append(nn.BatchNorm1d(self._net_cfg.hidden_dim, affine=False))
            layers.append(nn.ReLU())
        for _ in range(net_cfg.num_layers - 2):
            layers.append(nn.Linear(self._net_cfg.hidden_dim, self._net_cfg.hidden_dim))
            if net_cfg.use_activations:
                layers.append(nn.BatchNorm1d(self._net_cfg.hidden_dim, affine=False))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(self._net_cfg.hidden_dim, 10))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class QRandomClsModule(L.LightningModule):
    def __init__(self, model: nn.Module, learning_config: LearningConfig) -> None:
        super().__init__()
        self._model = model
        self._learn_cfg = learning_config

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return BinaryRandom(
            self._model.parameters(),
            max_value=1,
            step_size=1,
            lr=self._learn_cfg.learning_rate,
            min_prob=self._learn_cfg.min_prob,
            flip_factor=self._learn_cfg.flip_factor,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self._model(x["inputs"])
        return {"logits": outputs}

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch[0], batch[1]
        flip_mask = torch.rand(inputs.size(0), device=inputs.device) < 0.5
        if flip_mask.any():
            inputs = inputs.clone()
            inputs[flip_mask] = torch.flip(inputs[flip_mask], dims=[-1])
        return self.step((inputs, targets), "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        return self.step(batch, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch[0], batch[1]
        logits = self._model(inputs)
        logits_flip = self._model(torch.flip(inputs, dims=[-1]))
        avg_logits = (logits + logits_flip) / 2
        criterion = nn.CrossEntropyLoss()
        cel_loss = criterion(avg_logits, targets)
        p_correct = percent_correct(avg_logits, targets)
        self.log("test/cross_entropy_loss", cel_loss, prog_bar=True, sync_dist=True)
        self.log("test/percent_correct", p_correct, prog_bar=True, sync_dist=True)
        return cel_loss

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        step_type: Literal["train", "val", "test"],
    ) -> STEP_OUTPUT:
        inputs, targets = batch[0], batch[1]
        outputs = self.forward({"inputs": inputs})

        criterion = nn.CrossEntropyLoss()
        cel_loss = criterion(outputs["logits"], targets)
        p_correct = percent_correct(outputs["logits"], targets)
        self.log(
            f"{step_type}/cross_entropy_loss", cel_loss, prog_bar=True, sync_dist=True
        )
        self.log(
            f"{step_type}/percent_correct", p_correct, prog_bar=True, sync_dist=True
        )
        return cel_loss


def main() -> None:
    learning_config = LearningConfig()
    net_config = NetConfig()
    data_module = Cifar100Module(
        batch_size=learning_config.batch_size, num_workers=learning_config.num_workers
    )
    model = ConvNet()
    model = torch.compile(model)
    l_module = QRandomClsModule(model, learning_config)
    trainer = L.Trainer(
        max_epochs=10000,
        max_time="00:00:05:00",
        devices=1,
        log_every_n_steps=5,
        limit_val_batches=0,
        benchmark=True,
        precision="16-mixed",
    )
    trainer.fit(l_module, data_module)
    results = trainer.test(l_module, data_module)
    if results and trainer.is_global_zero:
        pct = results[0].get("test/percent_correct", 0.0)
        print(f"percent_correct: {pct:.4f}")
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"peak_vram_mb: {peak_vram_mb:.1f}")
        report_param_histogram(model)


def report_param_histogram(model: nn.Module) -> None:
    weight_counter: Counter[float] = Counter()
    bias_counter: Counter[float] = Counter()
    for name, param in model.named_parameters():
        unique, counts = torch.unique(param.data, return_counts=True)
        target = bias_counter if name.endswith("bias") else weight_counter
        for v, c in zip(unique.tolist(), counts.tolist()):
            target[v] += c

    ternary = {-1.0, 0.0, 1.0}
    ternary_w = {v: weight_counter.get(v, 0) for v in sorted(ternary)}
    ternary_b = {v: bias_counter.get(v, 0) for v in sorted(ternary)}
    non_ternary_w_count = sum(c for v, c in weight_counter.items() if v not in ternary)
    non_ternary_b_count = sum(c for v, c in bias_counter.items() if v not in ternary)

    print("\nTernary weight counts:")
    for v in sorted(ternary_w.keys()):
        print(f"  {v:+.1f}: {ternary_w[v]}")
    print("Ternary bias counts:")
    for v in sorted(ternary_b.keys()):
        print(f"  {v:+.1f}: {ternary_b[v]}")
    print(f"Non-ternary weights: {non_ternary_w_count}")
    print(f"Non-ternary biases: {non_ternary_b_count}")


if __name__ == "__main__":
    main()
