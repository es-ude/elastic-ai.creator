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
def precomputed(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.act_func"
    path2file = "verilog/precomputed.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("NUM_VALUES")
        .localparam("PRECOMPUTED")
        .add_module_name()
        .set_prototype("\n".join(read_text(package_path, path2file)))
        .build()
    )
    code = list()
    code.append((impl.name, _template.substitute(**impl.attributes)))
    return code
