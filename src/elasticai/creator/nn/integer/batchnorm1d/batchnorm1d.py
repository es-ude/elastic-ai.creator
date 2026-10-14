import copy

import torch
from torch import nn

from elasticai.creator.nn.integer.batchnorm1d.design import (
    BatchNorm1d as BatchNorm1dDesign,
)
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class BatchNorm1d(DesignCreatorModule, nn.BatchNorm1d):
    """Batch normalization with quantization-aware training and integer inference.

    Inputs have the shape `(batch, num_dimensions, norm_dim)`. Every feature
    is normalized with the mean and variance over the batch and the
    `num_dimensions` vectors, then scaled by `weight` and shifted by `bias`.

    `precompute` folds the running statistics into the parameters, so the
    integer path is an element-wise multiply-add per feature:
    `w' = w / sqrt(var + eps)` and `b' = b - w' * mean`.

    Note:
        `int_forward` currently deviates from `forward` by several
        quantization steps.
    """

    def __init__(
        self,
        *,
        name: str,
        norm_dim: int,
        in_features: int,
        out_features: int,
        num_dimensions: int,
        quant_bits: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
        enable_error_analysis: bool = False,
    ):
        """Create the layer.

        Args:
            name: Name of the layer, used for the saved quantization data.
            norm_dim: Number of normalized features, `num_features` of
                `nn.BatchNorm1d`.
            in_features: Features per input vector, passed to the design.
            out_features: Features per output vector, passed to the design.
            num_dimensions: Number of vectors per input.
            quant_bits: Bit width of inputs, outputs and weights.
            eps: Added to the variance for numerical stability.
            momentum: Weight of the current batch in the running statistics.
            affine: Passed to `nn.BatchNorm1d`; the layer needs weight and
                bias, so it must be `True`.
            track_running_stats: Passed to `nn.BatchNorm1d`.
            quant_data_dir: Directory for the quantized inputs and outputs
                saved by `int_forward`. `None` disables saving.
            device: Device of the quantization parameters.
            enable_error_analysis: Also save the float outputs of `forward`
                and the dequantized outputs of `int_forward`.
        """
        super().__init__(norm_dim, eps, momentum, affine, track_running_stats)
        self.num_dimensions = num_dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.norm_dim = norm_dim

        self.name = name
        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir
        self.enable_error_analysis = enable_error_analysis

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.modified_weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> BatchNorm1dDesign:
        """Create the VHDL design. Call `precompute` first.

        Args:
            name: Name of the VHDL entity.
        """
        return BatchNorm1dDesign(
            name=name,
            data_width=self.quant_bits,
            num_dimensions=self.num_dimensions,
            in_features=self.in_features,
            out_features=self.out_features,
            weights=self.q_weights.tolist(),
            bias=self.q_bias.tolist(),
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_w=self.modified_weight_QParams.zero_point.item(),
            z_b=self.q_bias_zero_point,
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _get_quantized_weights(
        self, weight: torch.Tensor, weight_QParams: nn.Module
    ) -> torch.Tensor:
        """Quantize `weight` with `weight_QParams` and subtract its zero point."""
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = weight_QParams.quantize(weight).to("cpu")

        if not weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                weight_QParams.zero_point,
                weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(
        self,
        bias: torch.Tensor,
        bias_QParams: QParams,
        given_quant_scale: torch.Tensor,
        given_quant_bits: int,
    ) -> tuple[torch.Tensor, QParams]:
        """Quantize `bias` with the given scale, zero point 0 and bit width.

        The bias is quantized with a copy of `bias_QParams`, so that
        `bias_QParams` keeps the parameters learned in training and `forward`
        gives the same outputs before and after `precompute`.

        Returns:
            The quantized bias and the QParams it was quantized with.
        """
        assert not self.training, "int_forward should be called in eval mode"

        q_bias_QParams = copy.deepcopy(bias_QParams)
        q_bias_QParams.set_scale_factor(given_quant_scale)
        q_bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        q_bias_QParams.set_quant_range(given_quant_bits)
        q_bias = q_bias_QParams.quantize(bias).to("cpu")
        return q_bias, q_bias_QParams

    def precompute(self):
        """Compute the integer parameters for `int_forward` and `create_design`.

        Folds the running statistics into weight and bias, quantizes both and
        approximates the requantization scale `M = s_x * s_w' / s_y` by
        `m_q * 2**-m_q_shift`. Must be called in eval mode, after training.
        """
        mean = self.running_mean
        var = self.running_var
        std = torch.sqrt(var + self.eps)

        self.tmp_quant_bits = (self.weight_QParams.quant_bits + 1) + (
            self.inputs_QParams.quant_bits + 1
        )

        weight = SimQuant.apply(self.weight, self.weight_QParams)
        modified_weight = weight / std
        self.modified_weight_QParams.update_quant_params(modified_weight)
        self.q_weights = self._get_quantized_weights(
            modified_weight, self.modified_weight_QParams
        )

        bias = SimQuant.apply(self.bias, self.bias_QParams)
        modified_bias = bias - modified_weight * mean
        new_bias_QParms_scale = (
            self.inputs_QParams.scale_factor * self.modified_weight_QParams.scale_factor
        )
        self.q_bias, q_bias_QParams = self._get_quantized_bias(
            bias=modified_bias,
            bias_QParams=self.bias_QParams,
            given_quant_scale=new_bias_QParms_scale,
            given_quant_bits=self.tmp_quant_bits,
        )
        # Only the zero point is kept. Storing the QParams module would add
        # its buffers to the state_dict of the layer.
        self.q_bias_zero_point = int(q_bias_QParams.zero_point.item())

        self.scale_factor_M = (
            self.inputs_QParams.scale_factor * self.modified_weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )

        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run integer-only inference.

        Computes `(q_x - z_x) * q_w' + q_b'` per feature, scales the result by
        `M` and adds `z_y`. Inputs and outputs are saved to `quant_data_dir`.

        Args:
            q_inputs: Inputs quantized with `inputs_QParams`, of shape
                `(batch, num_dimensions, norm_dim)`.

        Returns:
            The quantized outputs, with the shape of `q_inputs`.
        """
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )
        # integer-only
        tmp = self.math_ops.int_dotproduct(
            self.q_weights, q_inputs, self.tmp_quant_bits
        )

        tmp = self.math_ops.intadd(tmp, self.q_bias, self.tmp_quant_bits + 1)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        if self.enable_error_analysis:
            save_quant_data(
                self.outputs_QParams.dequantize(q_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
        return q_outputs

    def forward(
        self,
        inputs: torch.Tensor,
        given_inputs_QParams: object,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        """Normalize float inputs, optionally simulating quantization.

        In training mode, the batch statistics are used and the running
        statistics are updated; in eval mode, the running statistics are used.

        Args:
            inputs: Float inputs of shape `(batch, num_dimensions, norm_dim)`.
            given_inputs_QParams: Quantization parameters of the inputs,
                usually the output parameters of the previous layer, or
                `None`. If given in training mode, they replace
                `inputs_QParams` instead of being learned from `inputs`.
            enable_simquant: Simulate quantization of inputs, weight, bias
                and outputs.

        Returns:
            The float outputs, with the shape of `inputs`.
        """
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        if self.training:
            mean = inputs.mean(dim=(0, 1), keepdim=False)
            var = inputs.var(dim=(0, 1), keepdim=False)

            # Update the buffers in place and without autograd, so that they do
            # not keep the graph of every training step alive.
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    mean, alpha=self.momentum
                )
                self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        std = torch.sqrt(var + self.eps)

        if enable_simquant:
            if self.training:
                self.weight_QParams.update_quant_params(self.weight)
                self.bias_QParams.update_quant_params(self.bias)

            weight = SimQuant.apply(self.weight, self.weight_QParams)
            bias = SimQuant.apply(self.bias, self.bias_QParams)
        else:
            weight = self.weight
            bias = self.bias

        normed_inputs = (inputs - mean) / std
        outputs = weight * normed_inputs + bias

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)
            outputs = SimQuant.apply(outputs, self.outputs_QParams)
            if self.enable_error_analysis:
                save_quant_data(
                    outputs,
                    self.quant_data_dir,
                    f"{self.name}_y",
                )
        return outputs
