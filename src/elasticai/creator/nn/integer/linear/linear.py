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
        work_library_name: str = "work",
        resource_option: str = "distributed",
        weight_qparams_cls: QParamsFactory = AsymmetricSignedQParams,
        bias_qparams_cls: QParamsFactory = SymmetricSignedQParams,
        inputs_qparams_cls: QParamsFactory = AsymmetricSignedQParams,
        outputs_qparams_cls: QParamsFactory = AsymmetricSignedQParams,
    ):
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
            self.bias_QParams: QParams | None = bias_qparams_cls(
                quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
            ).to(device)
        else:
            self.bias_QParams = None
        self.inputs_QParams = inputs_qparams_cls(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = outputs_qparams_cls(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()

        # Integer parameters, set by `precompute`.
        self.precomputed = False
        self.q_weights: torch.Tensor | None = None
        self.q_bias: torch.Tensor | None = None
        self.scale_factor_M: torch.Tensor | None = None
        self.scale_factor_m_q: torch.Tensor | None = None
        self.scale_factor_m_q_shift: torch.Tensor | None = None

        self.use_parallelised_template = use_parallelised_template
        self.unroll_factor = unroll_factor
        self.use_pipeline_template = use_pipeline_template
        self.work_library_name = work_library_name
        self.resource_option = resource_option

    def create_design(self, name: str) -> LinearDesign:
        assert (
            self.precomputed
            and self.q_weights is not None
            and self.scale_factor_m_q is not None
            and self.scale_factor_m_q_shift is not None
        ), "precompute should be called before create_design"
        # The hardware uses one data width for inputs, weights and outputs, and
        # a bias width of 2 * (data_width + 1), which matches the bias bit
        # width (x_bits + 1) + (w_bits + 1) only if both are `quant_bits`.
        for element, qparams in [
            ("inputs", self.inputs_QParams),
            ("weights", self.weight_QParams),
            ("outputs", self.outputs_QParams),
        ]:
            if qparams.quant_bits != self.quant_bits:
                raise ValueError(
                    f"The hardware needs {element} with {self.quant_bits} bits, "
                    f"got {qparams.quant_bits}"
                )

        return LinearDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            num_dimensions=self.num_dimensions,
            out_features=self.out_features,
            weights=self.q_weights.tolist(),
            bias=self.q_bias.tolist() if self.q_bias is not None else [],
            m_q=int(self.scale_factor_m_q.item()),
            m_q_shift=int(self.scale_factor_m_q_shift.item()),
            z_x=int(self.inputs_QParams.zero_point.item()),
            z_w=int(self.weight_QParams.zero_point.item()),
            z_b=int(self.bias_QParams.zero_point.item())
            if self.bias_QParams is not None
            else 0,
            z_y=int(self.outputs_QParams.zero_point.item()),
            work_library_name=self.work_library_name,
            resource_option=self.resource_option,
            use_parallelised_template=self.use_parallelised_template,
            unroll_factor=self.unroll_factor,
            use_pipeline_template=self.use_pipeline_template,
        )

    def _get_quantized_weights(self) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = self.weight_QParams.quantize(self.weight).to("cpu")

        if not self.weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                self.weight_QParams.zero_point,
                self.weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(
        self, bias: torch.Tensor, bias_QParams: QParams
    ) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )
        new_bias_quant_bits = (self.inputs_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )
        bias_QParams.set_scale_factor(new_bias_scale_factor)
        bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        bias_QParams.set_quant_range(new_bias_quant_bits)
        q_bias = bias_QParams.quantize(bias).to("cpu")
        if not bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, bias_QParams.zero_point, new_bias_quant_bits + 1
            )
        return q_bias

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"
        self.q_weights = self._get_quantized_weights()
        if self.bias is not None and self.bias_QParams is not None:
            self.q_bias = self._get_quantized_bias(self.bias, self.bias_QParams)
        else:
            self.q_bias = None

        scale_factor_M = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor
        self.scale_factor_M = scale_factor_M
        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(scale_factor_M)
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert (
            self.precomputed
            and self.q_weights is not None
            and self.scale_factor_m_q is not None
            and self.scale_factor_m_q_shift is not None
        ), "precompute should be called before int_forward"
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
        check_dtype("inputs", inputs, *FLOAT_DTYPES)
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

                self.weight_QParams.update_quant_params(self.weight)
                if self.bias is not None and self.bias_QParams is not None:
                    self.bias_QParams.update_quant_params(self.bias)

            inputs = SimQuant.apply(inputs, self.inputs_QParams)
            weight = SimQuant.apply(self.weight, self.weight_QParams)
            bias = None
            if self.bias is not None and self.bias_QParams is not None:
                bias = SimQuant.apply(self.bias, self.bias_QParams)
        else:
            weight = self.weight
            bias = self.bias

        outputs = F.linear(inputs, weight, bias)

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
