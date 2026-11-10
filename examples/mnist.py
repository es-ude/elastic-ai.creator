import gzip
import urllib.request
from pathlib import Path
from copy import deepcopy
from shutil import rmtree

import numpy as np
import torch
import torch.nn as nn

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn import Sequential
from elasticai.creator.nn import fixed_point as nn_creator


BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def load_data(path2folder: Path, batch_size: int, do_minmax_norm: bool=True) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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

    scale = 1 / 255. if do_minmax_norm else 1.0

    x_train = scale * load_images(path2folder / FILES["train_images"])
    y_train = load_labels(path2folder / FILES["train_labels"])
    x_test = scale * load_images(path2folder / FILES["test_images"])
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


def patch_ap_types_for_macos(out_dir: Path) -> None:
    for name in ("ap_int_special.h", "ap_fixed_special.h"):
        header = out_dir / "firmware" / "ap_types" / name
        if header.exists():
            header.write_text("// stubbed out for the macOS C simulation build\n")


def convert_with_hls4ml(model: nn.Sequential, run_synthesis: bool=False, output_dir: str = "mlp_prj", backend: str="Vivado"):
    """You need Vivado/Vitis HLS on your client"""
    import hls4ml, sys

    if Path(output_dir).exists():
        rmtree(output_dir, ignore_errors=True)

    model_used = deepcopy(model).cpu().eval()
    config = hls4ml.utils.config_from_pytorch_model(
        model_used,
        input_shape=(14 * 14,),
        granularity="model",
        backend=backend,
        default_precision="ap_fixed<12,6>",
        default_reuse_factor=1,
    )
    config["Model"]["Strategy"] = "Latency"
    #  1 = fully parallel (max area, min latency). Larger values reuse multipliers,
    #  use a valid divisor of the layer's input width and switch to "Resource".
    config["Model"]["ReuseFactor"] = 400

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model_used,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        part="xc7s15ftgb196-2",
        io_type="io_parallel",
    )
    if sys.platform == "darwin":
        hls_model.write()
        patch_ap_types_for_macos(Path(output_dir))
        hls_model._compile()
    else:
        hls_model.compile()

    if run_synthesis:
        hls_model.build(csim=False, synth=True, export=True)
        hls4ml.report.read_vivado_report(output_dir)

    return hls_model


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

    # --- Export model
    path0 = OnDiskPath(name="hw_vhdl", parent=path2save.parent.as_posix())
    my_design = model_fxp.create_design("mnist")
    my_design.save_to(path0, take_vhdl=True)

    path1 = OnDiskPath(name="hw_verilog", parent=path2save.parent.as_posix())
    model_fxp.create_design("mnist").save_to(path1, take_vhdl=False)

    model_hls = convert_with_hls4ml(
        model=model_full,
        run_synthesis=False,
        output_dir="hw_hls4ml"
    )

    with torch.no_grad():
        x_test_dev = valid.dataset.tensors[0].to("cpu")
        y_test_dev = valid.dataset.tensors[1].to("cpu")
        frames = preprocess(x_test_dev).numpy()

        preds = torch.asarray(model_hls.predict(frames))
        acc_hls = (preds.argmax(dim=1) == y_test_dev).float().mean().item()
    print(f"\nTest-Genauigkeit: {acc_full:.4f} (full) - {acc_fxp:.4f} (fxp) - {acc_hls:.4f} (hls)")

