from typing import Literal

import torch

from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils.QParams import QParams

# NOTE: Import the functions from their modules, not from the package. This
#   module is imported while `quant_utils/__init__.py` is still running, so
#   `from elasticai.creator.nn.integer.quant_utils import scaling_M` would bind
#   the submodule `scaling_M.py` instead of the function.
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)


class MPQSupport:
    """Mixin that adds mixed-precision quantization (MPQ) support to a layer.

    With the `"inheritance"` strategy, a layer takes over the input `QParams`
    of the previous layer. With the `"requantizer"` strategy, it keeps its own
    input `QParams` and requantizes incoming integer values from the previous
    layer's parameters to its own.

    The layer must provide `name`, `quantizable_elements`, `math_ops` and a
    `<element>_QParams` attribute for each element.
    """

    name: str
    quantizable_elements: list
    math_ops: MathOperations

    def _init_mpq_attributes(
        self,
        MPQ_strategy: Literal["inheritance", "requantizer"] | None = None,
    ):
        self.MPQ_strategy = MPQ_strategy
        self.MPQ_elements = [
            element
            for element in self.quantizable_elements
            if element not in ["outputs", "weights", "bias"]
        ]
        for element in self.MPQ_elements:
            setattr(self, f"prev_{element}_QParams", None)

    def _check_quant_bits_configured(
        self, quant_bits_per_element: dict[str, int | None]
    ) -> None:
        """Raise an error if the config has no bit width for an element.

        The bias is skipped, because its bit width is derived from the inputs
        and weights.

        Args:
            quant_bits_per_element: Bit width per element as read from the
                config, `None` where the config has no entry.

        Raises:
            ValueError: If an element other than `"bias"` has no bit width.
        """
        missing = [
            f"{self.name}.{element}"
            for element, quant_bits in quant_bits_per_element.items()
            if quant_bits is None and element != "bias"
        ]
        if missing:
            raise ValueError(f"Missing quant_bits in config for: {', '.join(missing)}")

    def _handle_input_QParams(
        self,
        inputs: torch.Tensor,
        given_QParams: object,
        QParams_attr: str,
        prev_QParams_attr: str,
    ):
        if given_QParams is None:
            getattr(self, QParams_attr).update_quant_params(inputs)
            return

        if self.MPQ_strategy == "requantizer":
            getattr(self, QParams_attr).update_quant_params(inputs)
            if prev_QParams_attr is not None:
                setattr(self, prev_QParams_attr, given_QParams)
        elif self.MPQ_strategy == "inheritance":
            setattr(self, QParams_attr, given_QParams)
        else:
            raise ValueError(f"Unsupported MPQ strategy: {self.MPQ_strategy}")

    def _requantize_inputs(
        self,
        prev_q_inputs: torch.Tensor,
        prev_QParams: QParams,
        curr_QParams: QParams,
        m_q_1_shift: torch.Tensor,
        m_q_1: torch.Tensor,
    ) -> torch.Tensor:
        # s_curr (q_curr - z_curr) = s_prev  (q_prev - z_prev)
        curr_q_inputs = self.math_ops.intsub(
            prev_q_inputs,
            prev_QParams.zero_point,
            prev_QParams.quant_bits + 1,
        )

        # (q_curr - z_curr) = s_prev/s_curr (q_prev - z_prev)
        curr_q_inputs = simulate_bitshifting(
            curr_q_inputs,
            m_q_1_shift.to(curr_q_inputs.device),
            m_q_1.to(curr_q_inputs.device),
        )

        # q_curr = s_prev/s_curr (q_prev - z_prev) + z_curr
        curr_q_inputs = self.math_ops.intadd(
            curr_q_inputs,
            curr_QParams.zero_point,
            curr_QParams.quant_bits + 1,
        )
        return curr_q_inputs

    def _precompute_requantizer_params(self):
        if self.MPQ_strategy != "requantizer":
            return

        for element in self.MPQ_elements:
            prev_qparams_attr = f"prev_{element}_QParams"
            curr_qparams_attr = f"{element}_QParams"

            prev_qparams = getattr(self, prev_qparams_attr, None)
            curr_qparams = getattr(self, curr_qparams_attr)

            if prev_qparams is not None:
                requantizer_M = prev_qparams.scale_factor / curr_qparams.scale_factor
                m_q_shift, m_q = scaling_M(requantizer_M)
                setattr(self, f"requantizer_{element}_m_q_shift", m_q_shift)
                setattr(self, f"requantizer_{element}_m_q", m_q)

    def _should_use_requantizer(self, element):
        if self.MPQ_strategy != "requantizer":
            return False
        prev_qparams_attr = f"prev_{element}_QParams"
        return getattr(self, prev_qparams_attr, None) is not None

    def _apply_requantizer(self, q_inputs, element):
        if not self._should_use_requantizer(element):
            return q_inputs

        prev_qparams = getattr(self, f"prev_{element}_QParams")
        curr_qparams = getattr(self, f"{element}_QParams")
        m_q_shift = getattr(self, f"requantizer_{element}_m_q_shift")
        m_q = getattr(self, f"requantizer_{element}_m_q")

        return self._requantize_inputs(
            q_inputs,
            prev_qparams,
            curr_qparams,
            m_q_shift,
            m_q,
        )
