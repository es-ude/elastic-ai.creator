from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res


@type_handler_iterable
def uart(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.interface"
    code = list()

    _template = (
        TemplateDirector()
        .parameter("BITRATE")
        .parameter("BITWIDTH")
        .parameter("NSAMP")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/uart_com.v"))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))

    _fifo = (
        TemplateDirector()
        .parameter("FIFO_SIZE")
        .parameter("BITWIDTH")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/uart_fifo.v"))
        .build()
    )
    code.append(("uart_fifo", _fifo.substitute(impl.attributes)))

    _middleware = (
        TemplateDirector()
        .parameter("FIFO_SIZE")
        .parameter("BITWIDTH")
        .parameter("BITWIDTH_CMDS")
        .parameter("BITWIDTH_ADR")
        .parameter("BITWIDTH_DATA")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/uart_middleware.v"))
        .build()
    )
    code.append(("uart_middleware", _middleware.substitute(impl.attributes)))
    return code
