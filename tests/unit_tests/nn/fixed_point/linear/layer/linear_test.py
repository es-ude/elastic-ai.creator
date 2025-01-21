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
use ieee.numeric_std.all;               -- for type conversions

library work;
use work.all;

entity linear is -- layer_name is for distinguish same type of layers (with various weights) in one module
    generic (
        DATA_WIDTH   : integer := 16;
        FRAC_WIDTH   : integer := 8;
        X_ADDR_WIDTH : integer := 2;
        Y_ADDR_WIDTH : integer := 1;
        IN_FEATURE_NUM : integer := 3;
        OUT_FEATURE_NUM : integer := 2;
        RESOURCE_OPTION : string := "auto" -- can be "distributed", "block", or  "auto"
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_address : out std_logic_vector(X_ADDR_WIDTH-1 downto 0);
        y_address : in std_logic_vector(Y_ADDR_WIDTH-1 downto 0);

        x   : in std_logic_vector(DATA_WIDTH-1 downto 0);
        y  : out std_logic_vector(DATA_WIDTH-1 downto 0);

        done   : out std_logic
    );
end linear;

architecture rtl of linear is
    -----------------------------------------------------------
    -- Functions
    -----------------------------------------------------------
    -- macc
    function multiply_accumulate(w : in signed(DATA_WIDTH-1 downto 0);
                    x : in signed(DATA_WIDTH-1 downto 0);
                    y_0 : in signed(2*DATA_WIDTH-1 downto 0)
            ) return signed is

        variable TEMP : signed(DATA_WIDTH*2-1 downto 0) := (others=>'0');
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin
        TEMP := w * x;

        return TEMP+y_0;
    end function;

    function cut_down(x: in signed(2*DATA_WIDTH-1 downto 0))return signed is
        variable TEMP2 : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
        variable TEMP3 : signed(FRAC_WIDTH-1 downto 0) := (others=>'0');
    begin

        TEMP2 := x(DATA_WIDTH+FRAC_WIDTH-1 downto FRAC_WIDTH);
        TEMP3 := x(FRAC_WIDTH-1 downto 0);
        if TEMP2(DATA_WIDTH-1) = '1' and TEMP3 /= 0 then
            TEMP2 := TEMP2 + 1;
        end if;

        if x>0 and TEMP2<0 then
            TEMP2 := ('0', others => '1');
        elsif x<0 and TEMP2>0 then
            TEMP2 := ('1', others => '0');
        end if;
        return TEMP2;
    end function;

    -- Log2 function is for calculating the bitwidth of the address lines
    -- for bias and weights rom
    function log2(val : INTEGER) return natural is
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
    -- Signals
    -----------------------------------------------------------
    constant FXP_ZERO : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    constant FXP_ONE : signed(DATA_WIDTH-1 downto 0) := to_signed(2**FRAC_WIDTH,DATA_WIDTH);

    type t_state is (s_stop, s_forward, s_idle);

    signal n_clock : std_logic;
    signal w_in : std_logic_vector(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal b_in : std_logic_vector(DATA_WIDTH-1 downto 0) := (others=>'0');

    signal addr_w : std_logic_vector(log2(IN_FEATURE_NUM*OUT_FEATURE_NUM)-1 downto 0) := (others=>'0');
    --signal addr_b : std_logic_vector((log2(OUT_FEATURE_NUM)-1) downto 0) := (others=>'0');
    signal addr_b : std_logic_vector(Y_ADDR_WIDTH-1 downto 0) := (others=>'0');

    signal fxp_x, fxp_w, fxp_b, fxp_y : signed(DATA_WIDTH-1 downto 0) := (others=>'0');
    signal macc_sum : signed(2*DATA_WIDTH-1 downto 0) := (others=>'0');

    signal reset : std_logic := '0';
    signal state : t_state;

    -- simple solution for the output buffer
    type t_y_array is array (0 to OUT_FEATURE_NUM) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal y_ram : t_y_array;
    attribute rom_style : string;
    attribute rom_style of y_ram : signal is RESOURCE_OPTION;

begin

    -- connecting signals to ports
    n_clock <= not clock;

    fxp_w <= signed(w_in);
    fxp_x <= signed(x);
    fxp_b <= signed(b_in);

    -- connects ports
    reset <= not enable;

    linear_main : process (clock, enable, reset)
        variable current_neuron_idx : integer range 0 to OUT_FEATURE_NUM-1 := 0;
        variable current_input_idx : integer  range 0 to IN_FEATURE_NUM-1 := 0;
        variable var_addr_w : integer range 0 to OUT_FEATURE_NUM*IN_FEATURE_NUM-1 := 0;
        variable var_sum, var_y : signed(2*DATA_WIDTH-1 downto 0);
        variable var_w, var_x : signed(DATA_WIDTH-1 downto 0);
        variable y_write_en : std_logic;
        variable var_y_write_idx : integer;
    begin

        if (reset = '1') then
            state <= s_stop;
            done <= '0';

            current_neuron_idx := 0;
            current_input_idx := 0;
            var_addr_w := 0;

        elsif rising_edge(clock) then

            if state=s_stop then
                state <= s_forward;

                -- first add b accumulated sum
                var_y := (others=>'0');
                var_x := fxp_b;
                var_w := FXP_ONE;
            elsif state=s_forward then

                -- remapping to x and w
                var_y := macc_sum;
                var_x := fxp_x;
                var_w := fxp_w;

                if current_input_idx<IN_FEATURE_NUM-1 then
                    current_input_idx := current_input_idx + 1;
                    var_addr_w := var_addr_w + 1;
                else
                    current_input_idx := 0;

                    y_write_en := '1';
                    var_y_write_idx := current_neuron_idx;

                    if current_neuron_idx<OUT_FEATURE_NUM-1 then
                        current_neuron_idx := current_neuron_idx + 1;
                        var_addr_w := var_addr_w + 1;
                        state <= s_stop;
                    else
                        state <= s_idle;
                        done <= '1';
                    end if;

                end if;
            end if;

            var_sum := multiply_accumulate(var_w, var_x, var_y);
            macc_sum <= var_sum;

            if y_write_en='1'then
                y_ram(var_y_write_idx) <= std_logic_vector(cut_down(var_sum));
                y_write_en := '0';
            end if;

        end if;

        x_address <= std_logic_vector(to_unsigned(current_input_idx, x_address'length));
        addr_w <= std_logic_vector(to_unsigned(var_addr_w, addr_w'length));
        addr_b <= std_logic_vector(to_unsigned(current_neuron_idx, addr_b'length));
    end process linear_main;

    y_reading : process (clock, state)
    begin
        if (state=s_idle) or (state=s_stop) then
            if falling_edge(clock) then
                -- After the layer in at idle mode, y is readable
                -- but it only update at the rising edge of the clock
                y <= y_ram(to_integer(unsigned(y_address)));
            end if;
        end if;
    end process y_reading;

    -- Weights
    rom_w : entity work.linear_w_rom(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => addr_w,
        data => w_in
    );

    -- Bias
    rom_b : entity work.linear_b_rom(rtl)
    port map  (
        clk  => n_clock,
        en   => '1',
        addr => addr_b,
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
