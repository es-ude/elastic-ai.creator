from collections.abc import Iterable
from datetime import datetime

from elasticai.creator.file_generation.resource_utils import read_text
from elasticai.creator.hdl_ir import DataGraph
from elasticai.creator.ir2verilog import (
    Code,
    Registry,
    TemplateDirector,
    type_handler_iterable,
)


@type_handler_iterable()
def precomputed_lut(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.act_func"
    path2file = "verilog/precomputed_lut.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH_IN")
        .parameter("BITWIDTH_OUT")
        .localparam("NUM_VALUES")
        .localparam("VALUES_Y")
        .add_module_name()
        .set_prototype("\n".join(read_text(package_path, path2file)))
        .build()
    )
    code = list()
    code.append(
        (
            impl.name,
            _template.substitute(
                module_name=impl.attributes["name"].upper(),
                date_copy_created=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                **impl.attributes,
            ),
        )
    )
    return code


@type_handler_iterable()
def precomputed_scalar(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.act_func"
    path2file = "verilog/precomputed_scalar.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH_IN")
        .parameter("BITWIDTH_OUT")
        .localparam("NUM_VALUES")
        .localparam("VALUES_X")
        .localparam("VALUES_Y")
        .add_module_name()
        .set_prototype("\n".join(read_text(package_path, path2file)))
        .build()
    )
    code = list()
    code.append(
        (
            impl.name,
            _template.substitute(
                module_name=impl.attributes["name"].upper(),
                date_copy_created=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                **impl.attributes,
            ),
        )
    )
    return code
