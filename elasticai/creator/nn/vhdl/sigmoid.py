from elasticai.creator.base_modules.sigmoid import Sigmoid
from elasticai.creator.hdl.vhdl.designs.monotonic_increasing_scalar_function import (
    FPMonotonicIncModule,
)


class FPSigmoid(FPMonotonicIncModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-10, 10),
    ) -> None:
        super().__init__(
            base_module=Sigmoid(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
