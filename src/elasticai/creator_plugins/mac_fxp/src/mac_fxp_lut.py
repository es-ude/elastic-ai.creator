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
def mac_fxp_lut(impl: DataGraph, _: Registry) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.mac_fxp"
    path2file = "verilog/mac_fxp.v"

    _template = (
        TemplateDirector()
        .parameter("INPUT_BITWIDTH")
        .parameter("INPUT_NUM_DATA")
        .parameter("NUM_MULT_PARALLEL")
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
    # TODO: Add this code?
    # Adding MULT code (may get easier)
    # mult_impl = deepcopy(impl)
    # mult_impl.name = "mult_lut_signed"
    # mult_impl.type = "mult"
    # mult_impl.data["module_name"] = "MULT_LUT_SIGNED".upper()
    # mult_impl.data["type"] = "mult"
    # mult_impl.data["build_tb"] = False
    # mult_impl.data.update({"BITWIDTH": mult_impl.data["INPUT_BITWIDTH"]})
    # mult_design = mult_lut_signed(mult_impl)
    # code.extend(mult_design)
    return code
