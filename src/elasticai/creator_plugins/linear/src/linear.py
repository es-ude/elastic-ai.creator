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
def linear_array(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.linear"
    path2file = "verilog/linear_array.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("NUM_MULT")
        .parameter("BITS_SCALE_BIAS")
        .parameter("BITS_SCALE_DOUT")
        .parameter("SIZE_INPUT")
        .parameter("SIZE_OUTPUT")
        .localparam("BIAS")
        .localparam("WEIGHTS")
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
def linear_array_pipelined(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.linear"
    path2file = "verilog/linear_array_pipelined.v"

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .parameter("NUM_MULT")
        .parameter("BITS_SCALE_BIAS")
        .parameter("BITS_SCALE_DOUT")
        .parameter("SIZE_INPUT")
        .parameter("SIZE_OUTPUT")
        .replace_instance_name(
            "LINEAR_ARRAY", f"LINEAR_ARRAY_{impl.name.split('_')[-1]}"
        )
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
