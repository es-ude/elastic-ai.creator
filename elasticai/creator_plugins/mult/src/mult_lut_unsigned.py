from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res


@type_handler_iterable
def mult_lut_unsigned(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.mult"
    code = list()

    if impl.data['BITWIDTH'] in [6]:
        path2file = f'verilog/mult_dadda_u{impl.data['BITWIDTH']}.v'
        top_name = f'MULT_DADDA_UNSIGNED_{impl.data['BITWIDTH']}BIT'
    else:
        path2file = "verilog/mult_lut_unsigned.v"
        top_name = 'MULT_LUT_unSIGNED'

    _template = (
        TemplateDirector()
        .parameter("BITWIDTH")
        .add_module_name()
        .set_prototype(res.read_text(package_path, path2file))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))
    return code
