from elasticai.creator.hdl.code_generation.abstract_base_template import TemplateConfig, TemplateExpander, module_to_package

"""
Since we want to programmatically use designs that are either hand-written or based on hand-written templates
these designs need to adhere to a well defined protocol. As a hardware designer you specify your protocol
and the expected version of the elasticai.creator in a file called `design_meta.toml` that lives in the
same folder as your `layer.py` that defines how the hdl code for the hardware design and the behaviour of
the corresponding neural network layer in software.

To help you as hardware designer stick to specific protocol you can generate a base template, that you
can use as a starting point to develop your design.
"""

class BaseTemplateGenerator:
    def generate(self) -> str:
        return _generate_base_template_for_hw_block_protocol()


def _generate_base_template_for_hw_block_protocol() -> str:
    signals = """enable : in std_logic;
    clock  : in std_logic;
    x_address : out std_logic_vector($x_address_width-1 downto 0);
    y_address : in std_logic_vector($y_address_width-1 downto 0);

    x   : in std_logic_vector($x_width-1 downto 0);
    y  : out std_logic_vector($y_width-1 downto 0);

    done   : out std_logic"""
    config = TemplateConfig(
        package=module_to_package(_generate_base_template_for_hw_block_protocol.__module__),
        file_name="base_template.tpl.vhd",
        parameters={"signals": signals}
    )
    return "\n".join(TemplateExpander(config).lines())
    
