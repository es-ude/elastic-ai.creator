from itertools import chain
from typing import cast

import pytest
import torch

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
    to_hex,
)
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    generate_hex_for_rom,
)
from elasticai.creator.hdl.vhdl.designs.rom import Rom
from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.translatable_modules.vhdl.fp_linear_1d import FPLinear1d
from elasticai.creator.translatable_modules.vhdl.lstm.lstm import (
    FixedPointLSTMWithHardActivations as LSTM,
)
from elasticai.creator.translatable_modules.vhdl.lstm.lstm import LSTMNetwork


class ExpectedCode:
    def __init__(self, name: str):
        self.config = TemplateConfig(
            package=module_to_package(LSTM.__module__),
            file_name=name,
            parameters={},
        )

    def lines(self) -> list[str]:
        expander = TemplateExpander(config=self.config)
        return expander.lines()


class ExpectedLSTMNetworkCode(ExpectedCode):
    def __init__(self):
        super().__init__("lstm_network.tpl.vhd")


@pytest.mark.parametrize("total_bits", (8, 9, 10, 6))
def test_lstm_network_for_different_total_bits(total_bits):
    actual, expected = generate_lstm_network_and_expected_code(
        total_bits=total_bits, input_size=6, hidden_size=20, frac_bits=4
    )
    assert actual == expected


@pytest.mark.parametrize("input_size", (3, 4))
def test_lstm_network_for_different_input_sizes(input_size):
    actual, expected = generate_lstm_network_and_expected_code(
        total_bits=8, input_size=input_size, hidden_size=20, frac_bits=4
    )
    assert actual == expected


def generate_lstm_network_and_expected_code(
    total_bits: int, frac_bits: int, hidden_size: int, input_size: int
) -> tuple[list[str], list[str]]:
    model = LSTMNetwork(
        [
            LSTM(
                total_bits=total_bits,
                frac_bits=frac_bits,
                input_size=input_size,
                hidden_size=hidden_size,
            ),
            FPLinear1d(
                in_features=hidden_size,
                out_features=1,
                total_bits=total_bits,
                frac_bits=frac_bits,
                bias=True,
            ),
        ]
    )
    destination = InMemoryPath("lstm_network", parent=None)
    model.translate().save_to(destination)
    expected = ExpectedLSTMNetworkCode()
    expected.config.parameters.update(
        data_width=f"{total_bits}",
        frac_width=f"{frac_bits}",
        in_addr_width="4",
        input_size=f"{input_size}",
        hidden_size=f"{hidden_size}",
        x_h_addr_width=f"{calculate_address_width(hidden_size + input_size)}",
        hidden_addr_width=f"{calculate_address_width(hidden_size + input_size)}",
        w_addr_width=(
            f"{calculate_address_width((hidden_size + input_size) * hidden_size)}"
        ),
        linear_in_features="20",
        linear_out_features="1",
    )
    destination = InMemoryPath("lstm_network", parent=None)
    model.translate().save_to(destination)
    return destination["lstm_network"].text, expected.lines()


def test_lstm_cell_creates_lstm_cell_file(lstm_destination):
    expected = ExpectedCode("lstm_cell.tpl.vhd")
    total_bits = 16
    frac_bits = 8
    hidden_size = 20
    input_size = 6

    expected.config.parameters.update(
        library="work",
        data_width=f"{total_bits}",
        frac_width=f"{frac_bits}",
        input_size=f"{input_size}",
        hidden_size=f"{hidden_size}",
        hidden_addr_width="5",
        x_h_addr_width="5",
        w_addr_width=(
            f"{calculate_address_width((input_size + hidden_size) * hidden_size)}"
        ),
        name="lstm_cell",
        in_addr_width="4",
    )
    model = LSTMNetwork(
        [
            LSTM(
                total_bits=total_bits,
                frac_bits=frac_bits,
                input_size=input_size,
                hidden_size=hidden_size,
            ),
            FPLinear1d(
                in_features=hidden_size,
                out_features=1,
                total_bits=total_bits,
                frac_bits=frac_bits,
                bias=True,
            ),
        ]
    )
    design = model.translate()
    design.save_to(lstm_destination)
    lstm_cell_folder = cast(InMemoryPath, lstm_destination["lstm_cell"])
    lstm_cell_file = cast(InMemoryFile, lstm_cell_folder["lstm_cell"])
    actual = lstm_cell_file.text
    assert actual[0:60] == expected.lines()[0:60]


@pytest.fixture
def build_folder():
    return InMemoryPath(name="build", parent=None)


@pytest.fixture
def lstm_destination(build_folder):
    return build_folder.create_subpath("lstm")


def test_wi_rom_file_contains_32_zeros_for_input_size_1_and_hidden_size_4(
    lstm_destination,
):
    total_bits = 8
    input_size = 1
    hidden_size = 4
    destination = lstm_destination

    rom_address_width = calculate_address_width(
        (input_size + hidden_size) * hidden_size
    )
    expected_rom = Rom(
        name="rom_wi_lstm_cell",
        data_width=total_bits,
        values_as_integers=[0] * 2**rom_address_width,
    )
    destination_for_expected_rom = InMemoryPath("build", parent=None)
    expected_rom.save_to(destination_for_expected_rom.create_subpath("rom"))
    model = LSTM(
        total_bits=total_bits,
        frac_bits=4,
        input_size=input_size,
        hidden_size=hidden_size,
    )
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter *= 0
    design = model.translate()
    design.save_to(destination)

    actual = destination["wi_rom"].text
    expected = destination_for_expected_rom["rom"].text
    assert actual == expected


def test_wi_rom_file_contains_20_ones_and_12_zeros_for_input_size_1_and_hidden_size_4(
    lstm_destination,
):
    destination = lstm_destination
    total_bits = 8
    input_size = 1
    hidden_size = 4
    frac_bits = 4
    model = LSTM(
        total_bits=total_bits,
        frac_bits=frac_bits,
        input_size=input_size,
        hidden_size=hidden_size,
    )
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter *= 0
            parameter += torch.ones_like(parameter)
    design = model.translate()
    design.save_to(destination)
    rom_address_width = calculate_address_width(
        (input_size + hidden_size) * hidden_size
    )
    actual = cast(InMemoryFile, destination.children["wi_rom"]).text
    values = []
    for _ in range((input_size + hidden_size) * hidden_size):
        value_of_one_with_n_bits_for_fraction = 1 << frac_bits
        values.append(to_hex(value_of_one_with_n_bits_for_fraction, bit_width=8))
    for _ in range(2**rom_address_width - len(values)):
        values.append("00")
    expected = prepare_rom_file(values, rom_address_width, total_bits)
    assert actual == expected


def test_saves_all_necessary_subdesign_files(lstm_destination):
    total_bits = 8
    input_size = 1
    hidden_size = 4
    frac_bits = 4
    model = LSTM(
        total_bits=total_bits,
        frac_bits=frac_bits,
        input_size=input_size,
        hidden_size=hidden_size,
    )
    model.translate().save_to(lstm_destination)
    rom_suffixes = ("f", "i", "g", "o")
    weight_names = (f"w{suffix}" for suffix in rom_suffixes)
    biases_names = (f"b{suffix}" for suffix in rom_suffixes)
    rom_parameter_names = (f"{name}_rom" for name in chain(weight_names, biases_names))
    expected_file_names = [
        f"{name}.vhd"
        for name in chain(
            rom_parameter_names,
            (
                "hard_sigmoid",
                "hard_tanh",
                "lstm_cell_dual_port_2_clock_ram",
                "lstm_cell",
            ),
        )
    ]
    expected_file_names = sorted(expected_file_names)
    actual_file_names = []
    for child in lstm_destination.children.values():
        assert isinstance(child, InMemoryFile)
        actual_file_names.append(child.name)
    actual_file_names = sorted(actual_file_names)
    assert actual_file_names == expected_file_names


@pytest.fixture
def lstm_network_with_single_linear_layer():
    model = LSTMNetwork(
        [
            LSTM(
                total_bits=16,
                frac_bits=8,
                input_size=1,
                hidden_size=4,
            ),
            FPLinear1d(
                in_features=4, out_features=1, total_bits=16, frac_bits=8, bias=True
            ),
        ]
    )
    return model


def test_saves_linear_layer_files(
    lstm_destination, lstm_network_with_single_linear_layer
):
    model = lstm_network_with_single_linear_layer
    design = model.translate()
    design.save_to(lstm_destination)
    assert "fp_linear_1d_0" in lstm_destination.children


def test_connects_linear_layer(lstm_destination, lstm_network_with_single_linear_layer):
    model = lstm_network_with_single_linear_layer
    design = model.translate()
    design.save_to(lstm_destination)


def prepare_rom_file(values: list[str], rom_address_width, total_bits) -> list[str]:
    rom_value = ",".join(map(generate_hex_for_rom, values))
    params: dict[str, str | list[str]] = dict(
        rom_value=rom_value,
        rom_addr_bitwidth=str(rom_address_width),
        rom_data_bitwidth=str(total_bits),
        name="rom_wi_lstm_cell",
    )
    template = TemplateExpander(
        TemplateConfig(
            module_to_package(LSTM.__module__),
            file_name="rom.tpl.vhd",
            parameters=params,
        ),
    )
    return template.lines()
