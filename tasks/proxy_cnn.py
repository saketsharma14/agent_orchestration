"""Tiny CNN proxy task used to evaluate candidate optimizers cheaply.

Deliberately small: a few thousand MNIST examples, a 2-conv-layer CNN, a
handful of epochs. The goal is fast iteration during search, not SOTA
accuracy -- promising candidates get validated on a larger task separately
(e.g. a CIFAR-10 subset or a tiny transformer, once the harness works).
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SUBSET_SIZE = 4000
EVAL_SUBSET_SIZE = 1000
EPOCHS = 3
BATCH_SIZE = 64
TARGET_LOSS = 0.3  # used for the "steps-to-threshold" convergence metric


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc(x)


def _get_loaders():
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_ds = Subset(train_full, range(TRAIN_SUBSET_SIZE))
    eval_ds = Subset(test_full, range(EVAL_SUBSET_SIZE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)
    return train_loader, eval_loader


def train_and_score(optimizer_cls) -> dict:
    torch.manual_seed(0)
    model = TinyCNN().to(DEVICE)
    optimizer = optimizer_cls(model.parameters())
    train_loader, eval_loader = _get_loaders()

    step = 0
    steps_to_target = None
    start = time.time()

    for _epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            if not torch.isfinite(loss):
                return {
                    "diverged": True,
                    "final_loss": None,
                    "accuracy": None,
                    "steps_to_target": None,
                    "wall_time_s": time.time() - start,
                }
            loss.backward()
            optimizer.step()
            step += 1
            if steps_to_target is None and loss.item() < TARGET_LOSS:
                steps_to_target = step

    model.eval()
    correct, total = 0, 0
    final_loss = 0.0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            final_loss += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return {
        "diverged": False,
        "final_loss": final_loss / total,
        "accuracy": correct / total,
        "steps_to_target": steps_to_target,
        "wall_time_s": time.time() - start,
    }