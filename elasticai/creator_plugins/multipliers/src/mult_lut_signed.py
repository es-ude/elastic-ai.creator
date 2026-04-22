from collections.abc import Iterable

from elasticai.creator.file_generation.resource_utils import read_text
from elasticai.creator.hdl_ir import DataGraph
from elasticai.creator.ir2verilog import (
    Code,
    Registry,
    TemplateDirector,
    type_handler_iterable,
)


@type_handler_iterable()
def mult_lut_signed(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.multipliers"
    code = list()

    if impl.attributes["BITWIDTH"] in [2, 4, 6, 8, 10, 12]:
        path2file = f"verilog/mult_dadda_s{impl.attributes['BITWIDTH']}.v"
    else:
        path2file = "verilog/mult_lut_signed.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .add_module_name()
        .set_prototype("\n".join(read_text(package_path, path2file)))
        .build()
    )
    code.append((impl.name, _template.substitute(**impl.attributes)))
    return code
