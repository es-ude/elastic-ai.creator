import pytest
import torch
from torch.nn.functional import silu as torch_silu

from elasticai.creator.nn.fixed_point import quantize as fxp_quantize
from tests.tensor_test_case import assertTensorEqual

from .adaptable_silu import AdaptableSiLU


def fxp_args() -> dict[str, int]:
    return dict(total_bits=16, frac_bits=8)


def quantize(x: torch.Tensor) -> torch.Tensor:
    return fxp_quantize(x=x, **fxp_args())


def forge_silu(scale: float, beta: float) -> AdaptableSiLU:
    silu = AdaptableSiLU(num_steps=5, sampling_intervall=(-10, 10), **fxp_args())
    silu.load_state_dict(
        {
            "_base_module.scale": torch.tensor([scale]),
            "_base_module.beta": torch.tensor([beta]),
            "_step_lut": torch.linspace(-10, 10, 5),
        }
    )
    return silu


@pytest.mark.parametrize("scale,beta", [(1, 0), (-1, 0), (1, 1), (1, -1), (2, 3)])
def test_silu_with_several_scale_and_beta(scale: float, beta):
    silu = forge_silu(scale, beta)

    inputs = torch.tensor([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    actual_outputs = silu(inputs)

    target_inputs = torch.tensor([-10.0, -10.0, -5.0, 0.0, 5.0, 10.0, 10.0])
    target_outputs = quantize(scale * torch_silu(target_inputs) + beta)

    assertTensorEqual(target_outputs, actual_outputs)
