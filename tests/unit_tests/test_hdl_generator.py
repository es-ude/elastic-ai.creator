"""Tests for HDL generator abstraction layer.

This module tests the unified interface for VHDL and Verilog code generation,
ensuring that both implementations work correctly and produce valid HDL code.
"""

import pytest

from elasticai.creator.hdl_generator import HDLLanguage, create_generator
from elasticai.creator.hdl_ir import Node, Shape
from elasticai.creator.ir.ir_v2 import attribute
from elasticai.creator.ir2vhdl import factory as vhdl_factory


@pytest.fixture
def sample_node() -> Node:
    """Create a sample node for testing."""
    return vhdl_factory.node(
        "test_module",
        attribute(),
        type="test_type",
        implementation="test_impl",
        input_shape=Shape(8),
        output_shape=Shape(8),
    )


class TestVHDLSignalCreation:
    """Test VHDL signal creation through HDL abstraction."""

    def test_create_single_bit_signal(self) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        signal = gen.create_signal("clk")

        definition = tuple(signal.define())
        assert len(definition) == 1
        assert "signal clk : std_logic" in definition[0]

    def test_create_single_bit_signal_explicit_width(self) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        signal = gen.create_signal("clk", width=1)

        definition = tuple(signal.define())
        assert len(definition) == 1
        assert "signal clk : std_logic" in definition[0]

    def test_create_vector_signal(self) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        signal = gen.create_signal("data", width=8)

        definition = tuple(signal.define())
        assert len(definition) == 1
        assert "signal data : std_logic_vector(8 - 1 downto 0)" in definition[0]

    def test_create_null_signal(self) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        signal = gen.create_null_signal("input_port")

        definition = tuple(signal.define())
        assert len(definition) == 0

    def test_signal_name_property(self) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        signal = gen.create_signal("test_signal", width=4)

        assert signal.name == "test_signal"

    def test_make_instance_specific(self) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        signal = gen.create_signal("data", width=8)
        instance_signal = signal.make_instance_specific("my_instance")

        assert instance_signal.name == "data_my_instance"
        definition = tuple(instance_signal.define())
        assert "data_my_instance" in definition[0]


class TestVerilogWireCreation:
    """Test Verilog wire creation through HDL abstraction."""

    def test_create_single_bit_wire(self) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        wire = gen.create_signal("clk")

        definition = tuple(wire.define())
        assert len(definition) == 1
        assert definition[0] == "wire clk;"

    def test_create_single_bit_wire_explicit_width(self) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        wire = gen.create_signal("clk", width=1)

        definition = tuple(wire.define())
        assert len(definition) == 1
        assert definition[0] == "wire clk;"

    def test_create_vector_wire(self) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        wire = gen.create_signal("data", width=8)

        definition = tuple(wire.define())
        assert len(definition) == 1
        assert definition[0] == "wire [7:0] data;"

    def test_create_null_wire(self) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        wire = gen.create_null_signal("input_port")

        definition = tuple(wire.define())
        assert len(definition) == 0

    def test_wire_name_property(self) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        wire = gen.create_signal("test_wire", width=4)

        assert wire.name == "test_wire"

    def test_make_instance_specific(self) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        wire = gen.create_signal("data", width=8)
        instance_wire = wire.make_instance_specific("my_instance")

        assert instance_wire.name == "data_my_instance"
        definition = tuple(instance_wire.define())
        assert "data_my_instance" in definition[0]


class TestVHDLInstanceCreation:
    """Test VHDL entity instance creation through HDL abstraction."""

    def test_create_simple_instance(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        clk = gen.create_null_signal("clk")
        data_in = gen.create_signal("data_in", width=8)

        instance = gen.create_instance(
            node=sample_node,
            generics={"WIDTH": "8"},
            ports={"clk": clk, "data_in": data_in},
        )

        assert instance.name == "test_module"
        assert instance.implementation == "test_impl"

    def test_instance_signal_definitions(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        clk = gen.create_null_signal("clk")
        data_in = gen.create_signal("data_in", width=8)

        instance = gen.create_instance(
            node=sample_node,
            generics={},
            ports={"clk": clk, "data_in": data_in},
        )

        definitions = list(instance.define_signals())
        # clk is null-defined, so only data_in should be defined
        assert len(definitions) == 1
        assert "data_in_test_module" in definitions[0]

    def test_instance_instantiation_code(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        clk = gen.create_null_signal("clk")
        data_in = gen.create_signal("data_in", width=8)

        instance = gen.create_instance(
            node=sample_node,
            generics={"WIDTH": "8"},
            ports={"clk": clk, "data_in": data_in},
        )

        instantiation = list(instance.instantiate())

        # Check entity instantiation line
        assert any(
            "test_module: entity work.test_impl" in line for line in instantiation
        )

        # Check generic map
        assert any("generic map" in line for line in instantiation)
        assert any("WIDTH => 8" in line for line in instantiation)

        # Check port map
        assert any("port map" in line for line in instantiation)
        assert any("clk => clk" in line for line in instantiation)
        assert any("data_in => data_in_test_module" in line for line in instantiation)

    def test_instance_without_generics(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VHDL)
        clk = gen.create_null_signal("clk")

        instance = gen.create_instance(
            node=sample_node,
            ports={"clk": clk},
        )

        instantiation = list(instance.instantiate())

        # Should not have generic map when no generics provided
        assert not any("generic map" in line for line in instantiation)


class TestVerilogInstanceCreation:
    """Test Verilog module instance creation through HDL abstraction."""

    def test_create_simple_instance(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        clk = gen.create_null_signal("clk")
        data_in = gen.create_signal("data_in", width=8)

        instance = gen.create_instance(
            node=sample_node,
            generics={"WIDTH": "8"},
            ports={"clk": clk, "data_in": data_in},
        )

        assert instance.name == "test_module"
        assert instance.implementation == "test_impl"

    def test_instance_signal_definitions(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        clk = gen.create_null_signal("clk")
        data_in = gen.create_signal("data_in", width=8)

        instance = gen.create_instance(
            node=sample_node,
            generics={},
            ports={"clk": clk, "data_in": data_in},
        )

        definitions = list(instance.define_signals())
        # clk is null-defined, so only data_in should be defined
        assert len(definitions) == 1
        assert "data_in_test_module" in definitions[0]

    def test_instance_instantiation_code(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        clk = gen.create_null_signal("clk")
        data_in = gen.create_signal("data_in", width=8)

        instance = gen.create_instance(
            node=sample_node,
            generics={"WIDTH": "8"},
            ports={"clk": clk, "data_in": data_in},
        )

        instantiation = list(instance.instantiate())

        # Check module instantiation with parameters
        assert any("test_impl #(" in line for line in instantiation)
        assert any(".WIDTH(8)" in line for line in instantiation)

        # Check port connections
        assert any(".clk(clk)" in line for line in instantiation)
        assert any(".data_in(data_in_test_module)" in line for line in instantiation)

    def test_instance_without_parameters(self, sample_node: Node) -> None:
        gen = create_generator(HDLLanguage.VERILOG)
        clk = gen.create_null_signal("clk")

        instance = gen.create_instance(
            node=sample_node,
            ports={"clk": clk},
        )

        instantiation = list(instance.instantiate())

        # Should instantiate without parameters
        assert any("test_impl test_module (" in line for line in instantiation)


class TestTemplateDirector:
    """Test template director functionality."""

    @pytest.fixture
    def vhdl_prototype(self) -> str:
        return """
entity adder is
    generic (
        WIDTH : natural := 8
    );
    port (
        clk : in std_logic;
        a : in std_logic_vector(WIDTH-1 downto 0);
        b : in std_logic_vector(WIDTH-1 downto 0);
        sum : out std_logic_vector(WIDTH-1 downto 0)
    );
end entity adder;
"""

    def test_vhdl_template_creation(self, vhdl_prototype: str) -> None:
        gen = create_generator(HDLLanguage.VHDL)

        template = (
            gen.create_template_director()
            .set_prototype(vhdl_prototype)
            .add_parameter("WIDTH")
            .build()
        )

        # Test that template can substitute values
        result = template.substitute(entity="my_adder", WIDTH="16")

        assert "my_adder" in result
        assert "WIDTH : natural := 16" in result

    def test_verilog_template_creation(self) -> None:
        gen = create_generator(HDLLanguage.VERILOG)

        verilog_prototype = """
module adder #(
    parameter WIDTH = 8
) (
    input clk,
    input [WIDTH-1:0] a,
    input [WIDTH-1:0] b,
    output [WIDTH-1:0] sum
);
endmodule
"""

        # For Verilog, we need to use the add_module_name method
        director = gen.create_template_director()
        director.set_prototype(verilog_prototype)
        director.add_parameter("WIDTH")

        # Verilog director has add_module_name which we need to call
        # Since our protocol doesn't expose this, we'll access the underlying director
        from elasticai.creator.hdl_generator.verilog_impl import VerilogTemplateDirector

        if isinstance(director, VerilogTemplateDirector):
            director._director.add_module_name()

        template = director.build()

        # Test that template can substitute values
        result = template.substitute(module_name="my_adder", WIDTH="16")

        assert "my_adder" in result


class TestLanguageSelection:
    """Test language selection and factory behavior."""

    def test_vhdl_and_verilog_produce_different_code(self, sample_node: Node) -> None:
        vhdl_gen = create_generator(HDLLanguage.VHDL)
        verilog_gen = create_generator(HDLLanguage.VERILOG)

        # Create signals with both languages
        vhdl_signal = vhdl_gen.create_signal("data", width=8)
        verilog_signal = verilog_gen.create_signal("data", width=8)

        vhdl_def = tuple(vhdl_signal.define())[0]
        verilog_def = tuple(verilog_signal.define())[0]

        # They should produce different syntax
        assert "std_logic_vector" in vhdl_def
        assert "wire" in verilog_def
        assert vhdl_def != verilog_def


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_complete_vhdl_workflow(self) -> None:
        """Test a complete workflow: create signals, instance, and generate code."""
        gen = create_generator(HDLLanguage.VHDL)

        # Create node
        node = vhdl_factory.node(
            "fifo",
            attribute(),
            type="buffer",
            implementation="simple_fifo",
            input_shape=Shape(8),
            output_shape=Shape(8),
        )

        # Create signals
        clk = gen.create_null_signal("clk")
        rst = gen.create_null_signal("rst")
        data_in = gen.create_signal("fifo_data_in", width=8)
        data_out = gen.create_signal("fifo_data_out", width=8)
        write_enable = gen.create_signal("fifo_wr_en")
        read_enable = gen.create_signal("fifo_rd_en")

        # Create instance
        instance = gen.create_instance(
            node=node,
            generics={"DEPTH": "16", "WIDTH": "8"},
            ports={
                "clk": clk,
                "rst": rst,
                "data_in": data_in,
                "data_out": data_out,
                "wr_en": write_enable,
                "rd_en": read_enable,
            },
        )

        # Generate signal definitions
        signal_defs = list(instance.define_signals())
        assert len(signal_defs) > 0

        # Generate instantiation
        instantiation = list(instance.instantiate())
        assert len(instantiation) > 0

        # Verify structure
        full_code = "\n".join(signal_defs + [""] + instantiation)
        assert "signal" in full_code
        assert "entity work.simple_fifo" in full_code
        assert "DEPTH => 16" in full_code
        assert "WIDTH => 8" in full_code

    def test_complete_verilog_workflow(self) -> None:
        """Test a complete workflow for Verilog."""
        gen = create_generator(HDLLanguage.VERILOG)

        # Create node
        node = vhdl_factory.node(
            "fifo",
            attribute(),
            type="buffer",
            implementation="simple_fifo",
            input_shape=Shape(8),
            output_shape=Shape(8),
        )

        # Create wires
        clk = gen.create_null_signal("clk")
        rst = gen.create_null_signal("rst")
        data_in = gen.create_signal("fifo_data_in", width=8)
        data_out = gen.create_signal("fifo_data_out", width=8)
        write_enable = gen.create_signal("fifo_wr_en")
        read_enable = gen.create_signal("fifo_rd_en")

        # Create instance
        instance = gen.create_instance(
            node=node,
            generics={"DEPTH": "16", "WIDTH": "8"},
            ports={
                "clk": clk,
                "rst": rst,
                "data_in": data_in,
                "data_out": data_out,
                "wr_en": write_enable,
                "rd_en": read_enable,
            },
        )

        # Generate wire definitions
        wire_defs = list(instance.define_signals())
        assert len(wire_defs) > 0

        # Generate instantiation
        instantiation = list(instance.instantiate())
        assert len(instantiation) > 0

        # Verify structure
        full_code = "\n".join(wire_defs + [""] + instantiation)
        assert "wire" in full_code
        assert "simple_fifo" in full_code
        assert ".DEPTH(16)" in full_code
        assert ".WIDTH(8)" in full_code
