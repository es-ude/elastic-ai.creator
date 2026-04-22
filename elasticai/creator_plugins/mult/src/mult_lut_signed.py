from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res


@type_handler_iterable
def mult_lut_signed(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.mult"
    code = list()

    if impl.data['BITWIDTH'] in [2, 4, 6, 8, 10, 12]:
        path2file = f'verilog/mult_dadda_s{impl.data['BITWIDTH']}.v'
        top_name = f'MULT_DADDA_SIGNED_{impl.data['BITWIDTH']}BIT'
    else:
        path2file = "verilog/mult_lut_signed.v"
        top_name = 'MULT_LUT_SIGNED'

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .add_module_name()
        .set_prototype(res.read_text(package_path, path2file))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))
    return code
