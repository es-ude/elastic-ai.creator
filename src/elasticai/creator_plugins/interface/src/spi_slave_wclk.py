from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res


@type_handler_iterable
def spi_slave_with_clk(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.interface"
    code = list()

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("CPOL")
        .parameter("CPHA")
        .parameter("MSB")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/spi_slave_wclk.v"))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))

    _middleware = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("MSB")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/spi_middleware_fpga.v"))
        .build()
    )
    middleware_name = f"spi_middleware"
    middleware_attributes = impl.attributes | dict(module_name=middleware_name.upper())
    code.append((middleware_name, _template.substitute(middleware_attributes)))
    return code
