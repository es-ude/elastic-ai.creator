import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import click
import torch
from torch.nn import (
    Conv1d,
    Flatten,
    Linear,
    Sequential,
)
from torch.nn.functional import mse_loss
from torch.optim import Adam
from vunit import VUnit

from elasticai.creator.ir2vhdl import Ir2Vhdl, PluginLoader
from elasticai.creator.vhdl.skeleton_id import update_skeleton_id_in_build_dir
from elasticai.creator_plugins.lutron_filter import (
    Binarize,
    Torch2IrConverter,
)
from elasticai.creator_plugins.lutron_filter.lowering_passes.time_multiplexed import (
    load_plugins,
)
from elasticai.creator_plugins.lutron_filter.lowering_passes.time_multiplexed import (
    lower as TimeMultiplexed,
)


def save_as_json(registry, dir):
    os.makedirs(dir, exist_ok=True)
    for impl in registry:
        with open(dir / f"{impl.name}.json", "w") as f:
            json.dump(impl.asdict(), f, indent=1)


def save_low_ir(registry, LOW_LEVEL_DIR):
    save_as_json(registry, LOW_LEVEL_DIR)


def save_high_ir(registry, HIGH_LEVEL_DIR):
    save_as_json(registry, HIGH_LEVEL_DIR)


def save_vhdl(registry, VHDL_DIR):
    vhdl = Ir2Vhdl()
    loader = PluginLoader(vhdl)
    plugins = [
        "counter",
        "skeleton",
        "middleware",
        "combinatorial",
        "lutron",
        "shift_register",
        "sliding_window",
        "striding_shift_register",
        "padding",
    ]
    for p in plugins:
        loader.load_from_package(".".join(["elasticai.creator_plugins", p]))
    os.makedirs(VHDL_DIR, exist_ok=True)
    middleware_dir = VHDL_DIR / "middleware"
    os.makedirs(middleware_dir, exist_ok=True)
    for name, code in vhdl(registry):
        if name.removesuffix(".vhd") in (
            "icapInterface",
            "middleware",
            "spi_slave",
            "UserLogicInterface",
            "InterfaceStateMachine",
            "env5_reconfig_top",
        ):
            dest = middleware_dir
        else:
            dest = VHDL_DIR
        with open(dest / name, "w") as f:
            for line in code:
                f.write(line)
                f.write("\n")


def train(model):
    optimizer = Adam(model.parameters(), lr=0.1)
    x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32).view(
        [4, 2, 1]
    )
    y = torch.tensor([[-1], [1], [1], [-1]], dtype=torch.float32).view([4, 1, 1])
    for epoch in range(300):
        prediction = model(x).view([4, 1, 1])
        loss = mse_loss(prediction, y)
        loss.backward()
        print(
            "prediction: {}, epoch: {} loss: {}".format(
                prediction.tolist(), epoch, loss.item()
            )
        )
        optimizer.step()
        optimizer.zero_grad()
        if loss.item() == 0:
            break


def get_trained_conv_model():
    conv_model = Sequential(
        OrderedDict(
            conv1=Conv1d(2, 2, kernel_size=1),
            bin1=Binarize(),
            conv2=Conv1d(2, 1, kernel_size=1),
            bin2=Binarize(),
        )
    )
    train(conv_model)
    return conv_model


def get_trained_lin_model():
    lin_model = Sequential(
        OrderedDict(
            flatten0=Flatten(),
            lin0=Linear(2, 2),
            bin1=Binarize(),
            flatten=Flatten(),
            lin1=Linear(2, 1),
            bin2=Binarize(),
        )
    )
    train(lin_model)
    return lin_model


def generate_high_level_ir(model):
    converter = Torch2IrConverter()
    high_level_ir = converter.convert(model, input_shape=[1, 2, 1])
    return tuple(high_level_ir.values())


@click.group(chain=True)
def main():
    pass


ROOT = Path(".")

BUILD_DIR = ROOT / "build"
HIGH_LEVEL_DIR = BUILD_DIR / "hl_ir"
LOW_LEVEL_DIR = BUILD_DIR / "low_ir"
VHDL_DIR = BUILD_DIR / "vhdl"


@click.argument("vunit_args", nargs=-1)
@main.command()
def simulate(vunit_args: tuple[str, ...]):
    """Simulate the generated network and run some tests.

    use -- to separate vunit args from main.py args
    """
    vu = VUnit.from_argv(vunit_args)
    vu.add_vhdl_builtins()
    vu.add_library("elasticai", vhdl_standard="08")
    vu.add_source_files(
        "{}/*.vhd".format(str(VHDL_DIR)), library_name="elasticai", vhdl_standard="08"
    )
    vu.add_source_files(
        "xor_skeleton_tb.vhd", library_name="elasticai", vhdl_standard="08"
    )
    vu.main()


@main.command()
def generate():
    """train and generate the network.

    Will remove the content of the build directory.
    """
    torch.random.manual_seed(1)
    print("writing to ", ROOT.absolute())
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(exist_ok=False)
    model = get_trained_conv_model()

    high_level_ir = generate_high_level_ir(model)
    save_high_ir(high_level_ir, HIGH_LEVEL_DIR)
    load_plugins("time_multiplexed_sequential", "grouped_filter")
    low_level_ir = tuple(TimeMultiplexed(high_level_ir))
    save_low_ir(low_level_ir, LOW_LEVEL_DIR)
    save_vhdl(low_level_ir, VHDL_DIR)
    update_skeleton_id_in_build_dir(VHDL_DIR)


if __name__ == "__main__":
    main()
