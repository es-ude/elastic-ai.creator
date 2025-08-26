from dataclasses import dataclass
from typing import cast
from unittest import TestCase, main

import torch
from torch import Tensor, cat, tensor, testing, zeros
from torch.nn import BatchNorm1d, Linear, Sequential

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn import Sequential as SequentialCreator
from elasticai.creator.nn.fixed_point import (
    BatchNormedLinear as BatchNormedLinearCreator,
)
from elasticai.creator.nn.fixed_point import (
    Linear as LinearCreator,
)


@dataclass
class SettingsTest:
    data_repeat_num: int = 3
    size_input_layer: int = 3
    size_output_layer: int = 1
    quant_width_total: int = 12
    quant_width_frac: int = 8


RecommendedSettings = SettingsTest()


def generate_test_data(settings: SettingsTest = RecommendedSettings) -> Tensor:
    outputs = zeros((1, settings.size_input_layer)).repeat(settings.data_repeat_num, 1)
    for idx in range(settings.data_repeat_num):
        outputs = cat((outputs, tensor([[-1.5, 0.5, 1.5]])), dim=0)
        outputs = cat((outputs, tensor([[1.5, 0.5, -1.5]])), dim=0)
    return outputs


def generate_torch_linear(settings: SettingsTest = RecommendedSettings) -> list:
    linear_layer = Linear(
        settings.size_input_layer, settings.size_output_layer, bias=True
    )
    linear_layer.bias.data = tensor(0.5)
    linear_layer.weight.data = tensor([[0.5, 1.0, -0.5]])
    return [linear_layer]


def generate_creator_linear(settings: SettingsTest = RecommendedSettings) -> list:
    linear_layer = LinearCreator(
        in_features=settings.size_input_layer,
        out_features=settings.size_output_layer,
        total_bits=settings.quant_width_total,
        frac_bits=settings.quant_width_frac,
        bias=True,
    )
    linear_layer.bias.data = tensor(0.5)
    linear_layer.weight.data = tensor([[0.5, 1.0, -0.5]])
    return [linear_layer]


def generate_torch_batchlinear(settings: SettingsTest = RecommendedSettings) -> list:
    linear_layer = generate_torch_linear(settings)

    batch1d_layer = BatchNorm1d(settings.size_output_layer)
    batch1d_layer.bias.data = tensor([0.25])
    batch1d_layer.weight.data = tensor([0.75])
    linear_layer.append(batch1d_layer)
    return linear_layer


def generate_creator_batchlinear(settings: SettingsTest = RecommendedSettings) -> list:
    batch1d_layer = BatchNormedLinearCreator(
        in_features=settings.size_input_layer,
        out_features=settings.size_output_layer,
        total_bits=settings.quant_width_total,
        frac_bits=settings.quant_width_frac,
        bias=True,
        bn_affine=True,
    )
    batch1d_layer.lin_bias.data = tensor([[0.5]])
    batch1d_layer.lin_weight.data = tensor([[0.5, 1.0, -0.5]])

    batch1d_layer.bn_bias.data = tensor([0.25])
    batch1d_layer.bn_weight.data = tensor([0.75])
    return [batch1d_layer]


def build_torch_sequential(nn_layer: list, do_training: bool = False) -> Sequential:
    model_build = Sequential()
    for layer in nn_layer:
        model_build.append(layer)

    if do_training:
        model_build.train()
    else:
        model_build.eval()
    return model_build


def build_creator_sequential(
    nn_layer: list, do_training: bool = False
) -> SequentialCreator:
    model_build = SequentialCreator()
    for layer in nn_layer:
        model_build.append(layer)

    if do_training:
        model_build.train()
    else:
        model_build.eval()
    return model_build


class TestCreatorLinear(TestCase):
    model_torch = build_torch_sequential(generate_torch_linear())
    model_creator = build_creator_sequential(generate_creator_linear())
    data_in = generate_test_data()

    def test_data_input_size(self):
        testing.assert_close(self.data_in.size(1), RecommendedSettings.size_input_layer)

    def test_data_output_size_torch(self):
        testing.assert_close(
            self.model_torch(self.data_in).size(1),
            RecommendedSettings.size_output_layer,
        )

    def test_data_output_size_creator(self):
        testing.assert_close(
            self.model_creator(self.data_in).size(1),
            RecommendedSettings.size_output_layer,
        )

    def test_creator_linear(self):
        data_out_torch = self.model_torch(self.data_in)
        data_out_creator = self.model_creator(self.data_in)
        testing.assert_close(
            data_out_torch, data_out_creator, atol=1 / (2**8), rtol=1 / (2**8)
        )


def test_inference_of_multidimensional_data() -> None:
    linear = LinearCreator(
        total_bits=16, frac_bits=8, in_features=3, out_features=2, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data)

    inputs = torch.tensor([1.0, 2.0, 3.0])
    expected = [6.0, 6.0]
    actual = linear(inputs).tolist()

    assert expected == actual


def test_overflow_behaviour() -> None:
    linear = LinearCreator(
        total_bits=4, frac_bits=1, in_features=2, out_features=1, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data) * 2

    inputs = torch.tensor([2.5, -1.0])
    expected = [3.0]  # quantize(2.5 * 2 - 1.0 * 2)
    actual = linear(inputs).tolist()

    assert expected == actual


def test_underflow_behaviour() -> None:
    linear = LinearCreator(
        total_bits=4, frac_bits=1, in_features=1, out_features=1, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data) * 0.5

    inputs = torch.tensor([0.5])
    expected = [0.0]
    actual = linear(inputs).tolist()

    assert expected == actual


def test_bias_addition() -> None:
    linear = LinearCreator(
        total_bits=16, frac_bits=8, in_features=1, out_features=1, bias=True
    )
    linear.weight.data = torch.ones_like(linear.weight.data)
    linear.bias.data = torch.ones_like(linear.bias.data) * 2

    inputs = torch.tensor([3.0])
    expected = [5.0]
    actual = linear(inputs).tolist()

    assert expected == actual


def test_linear_layer_creates_correct_design() -> None:
    expected_linear_code = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.all;

-- layer_name is for distinguish same type of layers (with various weights) in one module
-- MAC operator with one multiplier
entity linear is
    generic (
        DATA_WIDTH   : integer := 16;
        FRAC_WIDTH   : integer := 8;
        X_ADDR_WIDTH : integer := 2;
        Y_ADDR_WIDTH : integer := 1;
        IN_FEATURE_NUM : integer := 3;
        OUT_FEATURE_NUM : integer := 2;
        RESOURCE_OPTION : string := "auto"
        -- can be "distributed", "block", or  "auto"
    );
    port (
        enable      : in    std_logic;
        clock       : in    std_logic;
        x_address   : out   std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address   : in    std_logic_vector(Y_ADDR_WIDTH-1 downto 0);
        x           : in    std_logic_vector(DATA_WIDTH-1 downto 0);
        y           : out   std_logic_vector(DATA_WIDTH-1 downto 0);
        done        : out   std_logic
    );
end linear;

architecture rtl of linear is
    -----------------------------------------------------------
    -- Functions
    -----------------------------------------------------------
    -- FXP_ROUNDING with clamping if range violation is available
    function FXP_ROUNDING(
        x0: in signed(2*DATA_WIDTH-1 downto 0)
    ) return signed is
        variable TEMP0 : signed(DATA_WIDTH-1 downto 0) := (others => '0');
        variable TEMP1 : signed(FRAC_WIDTH-1 downto 0) := (others => '0');
    begin
        TEMP0 := x0(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP1 := x0(FRAC_WIDTH-1 downto 0);

        if (x0(2*DATA_WIDTH-1) = '1' and TEMP0(DATA_WIDTH-1) = '0') then
            TEMP0 := ('1', others => '0');
        elsif (x0(2*DATA_WIDTH-1) = '0' and TEMP0(DATA_WIDTH-1) = '1') then
            TEMP0 := ('0', others => '1');
        else
            if TEMP0(DATA_WIDTH-1) = '1' and TEMP1 /= 0 then
                TEMP0 := TEMP0 + 1;
            end if;
        end if;

        return TEMP0;
    end function;

    -- log2 function is for calculating the bitwidth of the address lines
    function log2(
        val : INTEGER
    ) return natural is
        variable res : natural;
    begin
        for i in 1 to 31 loop
            if (val <= (2 ** i)) then
                res := i;
                exit;
            end if;
        end loop;
        return res;
    end function log2;

    -----------------------------------------------------------
    -- Process
    -----------------------------------------------------------
    signal addr_w   : unsigned(log2(IN_FEATURE_NUM*OUT_FEATURE_NUM)-1 downto 0) := (others => '0');
    signal addr_b   : unsigned(Y_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal addr_x   : unsigned(X_ADDR_WIDTH-1 downto 0) := (others => '0');

    signal w_in, b_in           : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    signal fxp_x, fxp_w, fxp_b  : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal buf_x, buf_w, buf_b  : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal mac_y                : signed(2*DATA_WIDTH-1 downto 0) := (others=>'0');
    signal enable_mac, reset_mac, done_int : std_logic;

    -- simple solution for the output buffer
    type t_y_array is array (0 to OUT_FEATURE_NUM) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_ram                    : t_y_array;
    attribute rom_style             : string;
    attribute rom_style of y_ram    : signal is RESOURCE_OPTION;
begin
    -- connecting signals to ports
    fxp_w <= signed(w_in);
    fxp_x <= signed(x);
    fxp_b <= signed(b_in);

    done <= done_int;
    x_address <= std_logic_vector(addr_x);
    enable_mac <= enable and not done_int;

    -- Pipelined MAC operator and saving into buffer
    mac: process(clock)
    begin
        if rising_edge(clock) then
            if (enable_mac = '0') then
                buf_x <= (others => '0');
                buf_w <= (others => '0');
                buf_b <= (others => '0');
                mac_y <= (others => '0');
            else
                if (reset_mac = '1') then
                    buf_x <= (others => '0');
                    buf_w <= (others => '0');
                    buf_b <= (others => '0');
                    mac_y <= (others => '0');
                    y_ram(to_integer(unsigned(addr_b))) <= std_logic_vector(FXP_ROUNDING(mac_y + buf_w * buf_x + SHIFT_LEFT(RESIZE(buf_b, 2*DATA_WIDTH), FRAC_WIDTH)));
                else
                    buf_x <= fxp_x;
                    buf_w <= fxp_w;
                    buf_b <= fxp_b;
                    mac_y <= mac_y + (buf_w * buf_x);
                end if;
            end if;
        end if;
    end process mac;

    -- Counter Operator for controlling the linear layer
    control : process (clock)
    begin
        if rising_edge(clock) then
            if (enable = '0') then
                done_int <= '0';
                addr_x <= (others => '0');
                addr_w <= (others => '0');
                addr_b <= (others => '0');
                reset_mac <= '0';
            else
                if (done_int <= '0') then
                    if (addr_x = IN_FEATURE_NUM-1) then
                        if (reset_mac = '0') then
                            reset_mac <= '1';
                        else
                            reset_mac <= '0';

                            addr_x <= (others => '0');
                            if (addr_b = OUT_FEATURE_NUM-1) then
                                addr_b <= (others => '0');
                                addr_w <= (others => '0');
                                done_int <= '1';
                            else
                                addr_b <= addr_b + 1;
                                addr_w <= addr_w + 1;
                                done_int <= '0';
                            end if;
                        end if;
                    else
                        done_int <= '0';
                        addr_x <= addr_x + 1;
                        addr_b <= addr_b;
                        addr_w <= addr_w + 1;
                    end if;
                else
                    done_int <= '1';
                    addr_x <= (others => '0');
                    addr_w <= (others => '0');
                    addr_b <= (others => '0');
                end if;
            end if;
        end if;
    end process control;

    -- Reading operator
    y_reading : process (clock)
    begin
        if rising_edge(clock) then
            if (done_int = '1') then
                y <= y_ram(to_integer(unsigned(y_address)));
            end if;
        end if;
    end process y_reading;

    -- Weights
    rom_w : entity work.linear_w_rom(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => std_logic_vector(addr_w),
        data => w_in
    );

    -- Bias
    rom_b : entity work.linear_b_rom(rtl)
    port map  (
        clk  => clock,
        en   => '1',
        addr => std_logic_vector(addr_b),
        data => b_in
    );
end architecture rtl;"""
    linear = LinearCreator(
        total_bits=16, frac_bits=8, in_features=3, out_features=2, bias=False
    )
    design = linear.create_design("linear")
    destination = InMemoryPath("linear", parent=None)
    design.save_to(destination)
    actual_linear_code = "\n".join(cast(InMemoryFile, destination["linear"]).text)
    assert expected_linear_code == actual_linear_code


if __name__ == "__main__":
    main()
