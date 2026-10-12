import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer._dtypes import FLOAT_DTYPES, check_dtype
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.linear.design import Linear as LinearDesign
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    QParams,
    QParamsFactory,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Linear(DesignCreatorModule, nn.Linear):
    """Linear layer with quantization-aware training and integer inference.

    The layer has three modes:

    - `forward` with `enable_simquant=True` simulates quantization of inputs,
      weights, bias and outputs. In training mode it also updates their
      quantization parameters.
    - `int_forward` runs integer-only inference on quantized inputs, as the
      hardware does. Call `precompute` first.
    - `create_design` creates the VHDL design. Call `precompute` first.
    """

    def __init__(
        self,
        *,
        name: str,
        in_features: int,
        out_features: int,
        quant_bits: int,
        bias: bool = False,
        num_dimensions: int | None = None,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
        enable_error_analysis: bool = False,
        use_parallelised_template: bool = False,
        unroll_factor: int = 1,
        use_pipeline_template: bool = False,
        weight_qparams_cls: QParamsFactory = AsymmetricSignedQParams,
        bias_qparams_cls: QParamsFactory = SymmetricSignedQParams,
        inputs_qparams_cls: QParamsFactory = AsymmetricSignedQParams,
        outputs_qparams_cls: QParamsFactory = AsymmetricSignedQParams,
    ):
        """Create the layer.

        Args:
            name: Name of the layer, used for the VHDL entity and the saved
                quantization data.
            in_features: Number of input features.
            out_features: Number of output features.
            quant_bits: Bit width of inputs, outputs and weights.
            bias: Whether the layer has a bias.
            num_dimensions: Number of input vectors for 2D inputs of shape
                `(num_dimensions, in_features)`. `None` for 1D inputs.
            quant_data_dir: Directory for the quantized inputs and outputs
                saved by `int_forward`. `None` disables saving.
            device: Device of the quantization parameters.
            enable_error_analysis: Also save the float outputs of `forward`
                and the dequantized outputs of `int_forward`.
            use_parallelised_template: See `LinearDesign`.
            unroll_factor: See `LinearDesign`.
            use_pipeline_template: See `LinearDesign`.
            weight_qparams_cls: Quantization scheme of the weights.
            bias_qparams_cls: Quantization scheme of the bias.
            inputs_qparams_cls: Quantization scheme of the inputs.
            outputs_qparams_cls: Quantization scheme of the outputs.
        """
        super().__init__(in_features, out_features, bias)
        self.num_dimensions = num_dimensions

        self.name = name
        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir
        self.enable_error_analysis = enable_error_analysis

        self.weight_QParams = weight_qparams_cls(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        if bias:
            self.bias_QParams = bias_qparams_cls(
                quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
            ).to(device)
        self.inputs_QParams = inputs_qparams_cls(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = outputs_qparams_cls(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

        self.use_parallelised_template = use_parallelised_template
        self.unroll_factor = unroll_factor
        self.use_pipeline_template = use_pipeline_template

    def create_design(self, name: str) -> LinearDesign:
        """Create the VHDL design from the precomputed integer parameters.

        Args:
            name: Name of the VHDL entity.

        Raises:
            RuntimeError: If `precompute` has not been called.
        """
        if not self.precomputed:
            raise RuntimeError("precompute must be called before create_design")

        # The hardware always adds a bias, so a layer without one gets a bias
        # of zeros.
        if self.q_bias is not None:
            bias = self.q_bias.tolist()
            z_b = int(self.bias_QParams.zero_point.item())
        else:
            bias = [0] * self.out_features
            z_b = 0

        return LinearDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            num_dimensions=self.num_dimensions,
            out_features=self.out_features,
            weights=self.q_weights.tolist(),
            bias=bias,
            m_q=int(self.scale_factor_m_q.item()),
            m_q_shift=int(self.scale_factor_m_q_shift.item()),
            z_x=int(self.inputs_QParams.zero_point.item()),
            z_w=int(self.weight_QParams.zero_point.item()),
            z_b=z_b,
            z_y=int(self.outputs_QParams.zero_point.item()),
            work_library_name="work",
            resource_option="distributed",
            use_parallelised_template=self.use_parallelised_template,
            unroll_factor=self.unroll_factor,
            use_pipeline_template=self.use_pipeline_template,
        )

    def _get_quantized_weights(self) -> torch.Tensor:
        """Quantize the weights and subtract their zero point."""
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = self.weight_QParams.quantize(self.weight).to("cpu")

        if not self.weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                self.weight_QParams.zero_point,
                self.weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(self) -> torch.Tensor:
        """Quantize the bias with the scale of the accumulator.

        The bias is added to the accumulator of `(x - z_x) @ (w - z_w).T`, so
        it uses the scale factor `s_x * s_w`, zero point 0 and the bit width
        `(x_bits + 1) + (w_bits + 1)`. This overwrites the quantization
        parameters of `bias_QParams`.
        """
        assert not self.training, "int_forward should be called in eval mode"
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )
        new_bias_quant_bits = (self.inputs_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )
        self.bias_QParams.set_scale_factor(new_bias_scale_factor)
        self.bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        self.bias_QParams.set_quant_range(new_bias_quant_bits)
        q_bias = self.bias_QParams.quantize(self.bias).to("cpu")
        if not self.bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, self.bias_QParams.zero_point, new_bias_quant_bits + 1
            )
        return q_bias

    def precompute(self) -> None:
        """Compute the integer parameters for `int_forward` and `create_design`.

        Quantizes weights and bias and approximates the requantization scale
        `M = s_x * s_w / s_y` by `m_q * 2**-m_q_shift`. Must be called in eval
        mode, after training.
        """
        assert not self.training, "int_forward should be called in eval mode"
        self.q_weights = self._get_quantized_weights()
        if self.bias is not None:
            self.q_bias = self._get_quantized_bias()
        else:
            self.q_bias = None

        self.scale_factor_M = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run integer-only inference, as the hardware does.

        Computes `y = ((x - z_x) @ (w - z_w).T + b) * M + z_y`, with `M`
        applied as a multiplication by `m_q` and a right shift by `m_q_shift`.
        Inputs and outputs are saved to `quant_data_dir`.

        Args:
            q_inputs: Inputs quantized with `inputs_QParams`, dtype
                `torch.int32`.

        Returns:
            The quantized outputs, dtype `torch.int32`.

        Raises:
            TypeError: If `q_inputs` is not a `torch.int32` tensor.
        """
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"
        check_dtype("q_inputs", q_inputs, torch.int32)

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )

        tmp = self.math_ops.int_mac(
            x=q_inputs,
            w=self.q_weights,
            b=self.q_bias,
            x_quant_bits=self.inputs_QParams.quant_bits,
            w_quant_bits=self.weight_QParams.quant_bits,
        )

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
        given_inputs_QParams: QParams | None = None,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        """Apply the layer to float inputs, optionally simulating quantization.

        Args:
            inputs: Float inputs of shape `(..., in_features)`.
            given_inputs_QParams: Quantization parameters of the inputs,
                usually the output parameters of the previous layer. If given
                in training mode, they replace `inputs_QParams` instead of
                being learned from `inputs`.
            enable_simquant: Simulate quantization of inputs, weights, bias
                and outputs. If `False`, the layer acts like `nn.Linear`.

        Returns:
            The float outputs of shape `(..., out_features)`.

        Raises:
            TypeError: If `inputs` does not have a floating-point dtype.
        """
        check_dtype("inputs", inputs, *FLOAT_DTYPES)
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

                self.weight_QParams.update_quant_params(self.weight)
                if self.bias is not None:
                    self.bias_QParams.update_quant_params(self.bias)

            inputs = SimQuant.apply(inputs, self.inputs_QParams)
            weight = SimQuant.apply(self.weight, self.weight_QParams)
            bias = None
            if self.bias is not None:
                bias = SimQuant.apply(self.bias, self.bias_QParams)
        else:
            weight = self.weight
            bias = self.bias

        outputs = (
            F.linear(inputs, weight, bias)
            if bias is not None
            else F.linear(inputs, weight)
        )

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
