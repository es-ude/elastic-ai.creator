from ._common_imports import (
    Design,
    FixedPointConfig,
    InProjectTemplate,
    Path,
    Port,
    Signal,
    module_to_package,
)


class FPHardTanh(Design):
    def __init__(self, name: str, total_bits: int, frac_bits: int) -> None:
        super().__init__(name=name)
        self._data_width = total_bits
        fp_config = FixedPointConfig(frac_bits=frac_bits, total_bits=total_bits)
        self._template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="fp_hard_tanh.tpl.vhd",
            parameters=dict(
                data_width=str(self._data_width),
                one=str(fp_config.as_integer(1)),
                minus_one=str(fp_config.as_integer(-1)),
            ),
        )

    def save_to(self, destination: Path):
        destination.as_file(".vhd").write(self._template)

    @property
    def port(self) -> Port:
        return Port(
            incoming=[Signal("x", self._data_width)],
            outgoing=[Signal("y", self._data_width)],
        )
