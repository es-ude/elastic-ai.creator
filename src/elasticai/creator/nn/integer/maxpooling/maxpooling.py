import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.maxpooling.design import (
    MaxPooling1d as MaxPooling1dDesign,
)
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class MaxPooling1d(DesignCreatorModule, nn.Module):
    def __init__(
        self,
        *,
        name: str,
        in_features: int,
        out_features: int,
        in_num_dimensions: int,
        out_num_dimensions: int,
        kernel_size: int,
        quant_bits: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        super().__init__()

        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.in_num_dimensions = in_num_dimensions
        self.out_num_dimensions = out_num_dimensions
        self.kernel_size = kernel_size

        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.scale_factor_Math_ops = MathOperations()

    def create_design(self, name: str) -> MaxPooling1dDesign:
        return MaxPooling1dDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            out_features=self.out_features,
            in_num_dimensions=self.in_num_dimensions,
            out_num_dimensions=self.out_num_dimensions,
            kernel_size=self.kernel_size,
            work_library_name="work",
            resource_option="auto",
        )

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        batch_size, channels, seq_len = q_inputs.shape
        kernel_size = self.kernel_size
        stride = self.kernel_size

        output_length = (seq_len - kernel_size) // stride + 1

        q_outputs = torch.empty(
            (batch_size, channels, output_length),
            dtype=torch.int32,
            device=q_inputs.device,
        )

        for i in range(output_length):
            start = i * stride
            end = start + kernel_size
            q_outputs[:, :, i] = q_inputs[:, :, start:end].max(dim=2)[0]

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def forward(
        self,
        inputs: torch.Tensor,
        given_inputs_QParams: QParams | None = None,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = F.max_pool1d(
            inputs, kernel_size=self.kernel_size, stride=self.kernel_size
        )

        self.outputs_QParams = self.inputs_QParams
        return outputs
