from brevitas.core.function_wrapper import InplaceTensorClampSte, TensorClamp
from brevitas.quant.base import (
    NarrowIntQuant,
    PerTensorConstScaling2bit,
    SignedBinaryClampedConst,
)
from brevitas.quant.solver import ActQuantSolver, BiasQuantSolver, WeightQuantSolver


class BinaryWeights(SignedBinaryClampedConst, WeightQuantSolver):
    """
    Quantize weights the same way as QTorch Binarize quantizer
    """

    tensor_clamp_impl = InplaceTensorClampSte
    scaling_const = 1


class BinaryBias(SignedBinaryClampedConst, BiasQuantSolver):
    """
    Quantize bias the same way as QTorch Binarize quantizer
    """

    tensor_clamp_impl = InplaceTensorClampSte
    scaling_const = 1
    requires_input_scale = False
    requires_input_bit_width = False


class BinaryActivation(SignedBinaryClampedConst, ActQuantSolver):
    """
    Quantize values the same way as QTorch Binarize quantizer
    """

    tensor_clamp_impl = TensorClamp
    min_val = -1.0
    max_val = 1.0


class TernaryWeights(NarrowIntQuant, PerTensorConstScaling2bit, WeightQuantSolver):
    """
    Quantize weights the same way as QTorch Ternarize quantizer
    with a zero_window_width of 0.5 and without widening_factor
    """

    tensor_clamp_impl = InplaceTensorClampSte
    scaling_const = 1


class TernaryBias(NarrowIntQuant, PerTensorConstScaling2bit, BiasQuantSolver):
    """
    Quantize bias the same way as QTorch Ternarize quantizer
    with a zero_window_width of 0.5 and without widening_factor
    """

    tensor_clamp_impl = InplaceTensorClampSte
    scaling_const = 1
    requires_input_scale = False
    requires_input_bit_width = False


class TernaryActivation(NarrowIntQuant, PerTensorConstScaling2bit, ActQuantSolver):
    """
    Quantize values the same way as QTorch Ternarize quantizer
    with a zero_window_width of 0.5 and without widening_factor
    """

    tensor_clamp_impl = TensorClamp
    min_val = -1.0
    max_val = 1.0
