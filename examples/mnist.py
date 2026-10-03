import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.nn import fixed_point as nn_creator
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5


BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def load_data(path2folder: Path, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    path2folder.mkdir(parents=True, exist_ok=True)
    for name, filename in FILES.items():
        path = path2folder / filename
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL + filename, path)

    def load_images(path):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(path):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data

    x_train = load_images(path2folder / FILES["train_images"])
    y_train = load_labels(path2folder / FILES["train_labels"])
    x_test = load_images(path2folder / FILES["test_images"])
    y_test = load_labels(path2folder / FILES["test_labels"])

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    valid_ds = torch.utils.data.TensorDataset(x_test, y_test)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True),
    )


if __name__ == "__main__":
    path2save = find_project_root() / "examples" / "mnist_data"
    train, valid = load_data(
        path2folder=path2save,
        batch_size=1024
    )
    EPOCHS = 10
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    preprocess = nn.Sequential(
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten()
    )
    model_full = nn.Sequential(
        nn.Linear(14*14, 40),
        nn.LeakyReLU(),
        nn.Linear(40, 10),
    )
    model_fxp = Sequential(
        nn_creator.Linear(14*14, 40, total_bits=8, frac_bits=6),
        nn_creator.LeakyReLU2(total_bits=8, frac_bits=6),
        nn_creator.Linear(40, 10, total_bits=8, frac_bits=6),
    )

    preprocess.to(device)
    model_full.to(device)
    model_fxp.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)
    optimizer_fxp = torch.optim.Adam(model_fxp.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        preprocess.eval()
        model_full.train()

        total_loss_full = 0.0
        total_loss_fxp = 0.0
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device)
            frames = preprocess(xb)

            optimizer_full.zero_grad()
            preds_full = model_full(frames)
            loss_full = loss_fn(preds_full, yb)
            loss_full.backward()
            optimizer_full.step()

            optimizer_fxp.zero_grad()
            preds_fxp = model_fxp(frames)
            loss_fxp = loss_fn(preds_fxp, yb)
            loss_fxp.backward()
            optimizer_fxp.step()

            total_loss_full += loss_full.item() * xb.size(0)
            total_loss_fxp += loss_fxp.item() * xb.size(0)

        avg_loss_full = total_loss_full / len(train.dataset)
        avg_loss_fxp = total_loss_fxp / len(train.dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss_full: {avg_loss_full:.4f} - loss_fxp: {avg_loss_fxp:.4f}")

    model_full.eval()
    model_fxp.eval()
    with torch.no_grad():
        x_test_dev = valid.dataset.tensors[0].to(device)
        y_test_dev = valid.dataset.tensors[1].to(device)
        frames = preprocess(x_test_dev)

        preds = model_full(frames)
        acc_full = (preds.argmax(dim=1) == y_test_dev).float().mean().item()
        preds = model_fxp(frames)
        acc_fxp = (preds.argmax(dim=1) == y_test_dev).float().mean().item()
    print(f"\nTest-Genauigkeit: {acc_full:.4f} (full) - {acc_fxp:.4f} (fxp)")


    path0 = OnDiskPath(name="hw_vhdl", parent=path2save.parent.as_posix())
    my_design = model_fxp.create_design("mnist")
    my_design.save_to(path0, take_vhdl=True)
    firmware = FirmwareENv5(
        network=my_design,
        x_num_values=144,
        y_num_values=10,
        id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        skeleton_version="v2",
    )
    firmware.save_to(path0)

    path1 = OnDiskPath(name="hw_verilog", parent=path2save.parent.as_posix())
    model_fxp.create_design("mnist").save_to(path1, take_vhdl=False)
