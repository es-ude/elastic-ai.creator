from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res


@type_handler_iterable
def spi_master(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.interface"
    code = list()

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("CPOL")
        .parameter("CPHA")
        .parameter("MSB")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/spi_master.v"))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))
    return code
