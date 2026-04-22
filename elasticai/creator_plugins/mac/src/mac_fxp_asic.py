from collections.abc import Iterable
from copy import deepcopy
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res
from elasticai.creator_plugins.mult.src.mult_lut_signed import mult_lut_signed


@type_handler_iterable
def mac_fxp_asic(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.mac"
    code = list()

    _template = (
        TemplateDirector()
        .parameter("INPUT_BITWIDTH")
        .parameter("INPUT_NUM_DATA")
        .parameter("NUM_MULT_PARALLEL")
        .add_module_name()
        .set_prototype(res.read_text(package_path, f"verilog/mac_fxp_lut.v"))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))
    # Adding MULT code (may get easier)
    mult_impl = deepcopy(impl)
    mult_impl.name = "mult_lut_signed"
    mult_impl.type = "mult"
    mult_impl.data["module_name"] = "MULT_LUT_SIGNED".upper()
    mult_impl.data["type"] = "mult"
    mult_impl.data["build_tb"] = False
    mult_impl.data.update({"BITWIDTH": mult_impl.data["INPUT_BITWIDTH"]})
    mult_design = mult_lut_signed(mult_impl)
    code.extend(mult_design)
    return code
