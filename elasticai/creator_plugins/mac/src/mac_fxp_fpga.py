from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res


@type_handler_iterable
def mac_fxp_fpga(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.mac"
    code = list()

    _template = (
        TemplateDirector()
        .parameter("INPUT_BITWIDTH")
        .parameter("INPUT_NUM_DATA")
        .parameter("NUM_MULT_PARALLEL")
        .add_module_name()
        .set_prototype(res.read_text(package_path, f"verilog/mac_fxp_dsp.v"))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))

    return code
