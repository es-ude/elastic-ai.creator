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
)
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    generate_hex_for_rom,
)
from elasticai.creator.in_memory_path import (
    InMemoryFile,
    InMemoryPath,
    InMemoryPathForTesting,
)
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
            )
        ]
    )
    destination = InMemoryPathForTesting("lstm_network")
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
    destination = InMemoryPathForTesting("lstm_network")
    model.translate().save_to(destination)
    return destination.text, expected.lines()


def test_lstm_cell_creates_lstm_cell_file():
    expected = ExpectedCode("lstm_cell.tpl.vhd")
    total_bits = 16
    frac_bits = 8
    hidden_size = 20
    input_size = 6

    expected.config.parameters.update(
        libary="work",
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
    build_folder = InMemoryPath(name="build", parent=None)
    destination = InMemoryPath(name="lstm_network", parent=build_folder)
    model = LSTMNetwork(
        [
            LSTM(
                total_bits=total_bits,
                frac_bits=frac_bits,
                input_size=input_size,
                hidden_size=hidden_size,
            )
        ]
    )
    design = model.translate()
    design.save_to(destination)

    actual = cast(InMemoryFile, destination.children["lstm_cell"]).text
    assert actual[0:60] == expected.lines()[0:60]


def test_wi_rom_file_contains_32_zeros_for_input_size_1_and_hidden_size_4():
    build_folder = InMemoryPath(name="build", parent=None)
    destination = build_folder.create_subpath("lstm_cell")
    total_bits = 8
    input_size = 1
    hidden_size = 4

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
    rom_address_width = calculate_address_width(
        (input_size + hidden_size) * hidden_size
    )
    actual = cast(InMemoryFile, destination.children["wi_rom"]).text
    expected = prepare_rom_file(
        ["00"] * 2**rom_address_width, rom_address_width, total_bits
    )
    assert actual == expected


def prepare_rom_file(values: list[str], rom_address_width, total_bits) -> list[str]:
    rom_value = ",".join(map(generate_hex_for_rom, values))
    params = dict(
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
