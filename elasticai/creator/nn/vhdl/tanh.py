from elasticai.creator.base_modules.tanh import Tanh
from elasticai.creator.hdl.vhdl.designs.monotonic_increasing_scalar_function import (
    FPMonotonicIncModule,
)


class FPTanh(FPMonotonicIncModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-5, 5),
    ) -> None:
        super().__init__(
            base_module=Tanh(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
