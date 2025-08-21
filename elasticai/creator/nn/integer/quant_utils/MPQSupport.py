import torch

from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import scaling_M, simulate_bitshifting


class MPQSupport:
    def _init_mpq_attributes(self, **kwargs):
        self.MPQ_strategy = kwargs.get("MPQ_strategy")  # ["inheritance", "requantizer"]
        self.MPQ_elements = [
            element
            for element in self.quantizable_elements
            if element not in ["outputs", "weights", "bias"]
        ]
        for element in self.MPQ_elements:
            setattr(self, f"prev_{element}_QParams", None)

    def _handle_input_QParams(
        self,
        inputs: torch.FloatTensor,
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
        prev_q_inputs: torch.IntTensor,
        prev_QParams: object,
        curr_QParams: object,
        m_q_1_shift: int,
        m_q_1: int,
    ) -> torch.IntTensor:
        # s_curr (q_curr - z_curr) = s_prev  (q_prev - z_prev)
        curr_q_inputs = self.math_ops.intsub(
            prev_q_inputs,
            prev_QParams.zero_point,
            prev_QParams.quant_bits + 1,
        )

        # (q_curr - z_curr) = s_prev/s_curr (q_prev - z_prev)
        curr_q_inputs = simulate_bitshifting(
            curr_q_inputs,
            m_q_1_shift,
            m_q_1,
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
