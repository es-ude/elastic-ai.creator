from collections.abc import Iterable
from elasticai.creator.ir2verilog import (
    type_handler_iterable,
    Implementation,
    Code,
    TemplateDirector,
)
from importlib import resources as res


@type_handler_iterable
def modulator_pwm(impl: Implementation) -> Iterable[Code]:
    package_path = "elasticai.creator_plugins.modulator"
    code = list()

    _template = (
        TemplateDirector()
        .parameter("PERIOD_NUM_CYCLE")
        .add_module_name()
        .set_prototype(res.read_text(package_path, "verilog/modulator_pwm.v"))
        .build()
    )
    code.append((impl.name, _template.substitute(impl.attributes)))

    if impl.data['build_tb']:
        _testbench = (
            TemplateDirector()
            .localparam("PERIOD_NUM_CYCLE")
            .add_module_name()
            .replace_instance_name("PULSE_WIDTH_MODULATOR", impl.name.upper())
            .set_prototype(res.read_text(package_path, "verilog/modulator_pwm_tb.v"))
            .build()
        )
        tb_name = f"{impl.name}_tb"
        tb_attributes = impl.attributes | dict(module_name=tb_name.upper())
        code.append((tb_name, _testbench.substitute(tb_attributes)))
    return code
