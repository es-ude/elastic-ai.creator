# -*- coding: utf-8 -*-
"""Basic Qtorch application"""
from functools import partial

import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from elasticai.creator.qat.constraints import WeightClipper
from elasticai.creator.qat.layers import Binarize, QConv2d, QLinear

xy_train = torchvision.datasets.FashionMNIST(
    root="data/",
    download=True,
)

(x_train, y_train) = xy_train.data, xy_train.targets
x_train = x_train.float() / 255.0
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
xy_train = TensorDataset(x_train, y_train)

xy_valid = torchvision.datasets.FashionMNIST(
    train=False,
    root="data/",
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(mean), std=(std)),
        ]
    ),
    download=True,
)

print("mean: ", mean)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


convolutions = []


def Conv2d(**kwargs):
    layer = QConv2d(**kwargs)
    convolutions.append(layer)
    return layer


def Linear(**kwargs):
    layer = QLinear(**kwargs)
    convolutions.append(layer)
    return layer


def get_model():
    _Conv2d = partial(Conv2d, quantizer=Binarize(), bias=False)
    # _Conv2d = nn.Conv2d
    num_bits = 1
    activation = Binarize
    fc = partial(
        Linear, quantizer=Binarize(), bias=False, constraints=[WeightClipper()]
    )
    # fc = nn.Linear
    model = nn.Sequential(
        Lambda(lambda x: x.view(-1, 1, 28, 28)),
        _Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            constraints=[WeightClipper()],
        ),
        nn.BatchNorm2d(32),
        activation(),
        _Conv2d(
            in_channels=num_bits * 32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            constraints=[WeightClipper()],
        ),
        nn.BatchNorm2d(32),
        activation(),
        _Conv2d(
            in_channels=num_bits * 32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            constraints=[WeightClipper()],
        ),
        nn.BatchNorm2d(16),
        activation(),
        Lambda(lambda x: x.view(x.size(0), -1)),
        fc(in_features=256 * num_bits, out_features=10),
    )
    return model, optim.Adamax(model.parameters(), lr=0.001)


model, optimizer = get_model()


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def apply_weights_constraint():
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "constraints"):
                module.apply_constraint()


def fit(model, optimizer, loss_func, batch_size, epochs, train_dataset, valid_dataset):
    data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2)
    for epoch in range(epochs):
        model.train()
        for xb, yb in data_loader:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            optimizer.step()
            apply_weights_constraint()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_loader)
            valid_accuracy = sum(accuracy(model(xb), yb) for xb, yb in valid_loader)
            train_accuracy = sum(accuracy(model(xb), yb) for xb, yb in data_loader)

        print(
            epoch,
            " :",
            valid_loss / len(valid_loader),
            ", ",
            valid_accuracy / len(valid_loader),
            " train acc: ",
            train_accuracy / len(data_loader),
        )


fit(
    model=model,
    optimizer=optimizer,
    loss_func=F.cross_entropy,
    batch_size=256,
    epochs=30,
    train_dataset=xy_train,
    valid_dataset=xy_valid,
)
