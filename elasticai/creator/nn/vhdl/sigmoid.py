from elasticai.creator.base_modules.sigmoid import Sigmoid
from elasticai.creator.hdl.vhdl.designs.monotonously_increasing_precomputed_scalar_function.fp_monotonously_increasing_module import (
    FPMonotonouslyIncreasingModule,
)


class FPSigmoid(FPMonotonouslyIncreasingModule):
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
