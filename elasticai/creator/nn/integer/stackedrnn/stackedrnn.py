import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.rnnlayer import RNNLayer
from elasticai.creator.nn.integer.stackedrnn.design import (
    StackedRNN as StackedRNNDesign,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class StackedRNN(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.cell_type = kwargs.get("cell_type")
        self.num_layers = kwargs.get("num_layers")

        self.inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.window_size = kwargs.get("window_size")
        self.batch_size = kwargs.get("batch_size")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.rnn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.rnn_layers.append(
                RNNLayer(
                    inputs_size=self.inputs_size if i == 0 else self.hidden_size,
                    hidden_size=self.hidden_size,
                    quant_bits=self.quant_bits,
                    window_size=self.window_size,
                    cell_type=self.cell_type,
                    batch_size=self.batch_size,
                    name=self.name + f"_rnn_layer_{i}",
                    quant_data_dir=self.quant_data_dir,
                    device=device,
                )
            )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_prev_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.c_prev_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name) -> StackedRNNDesign:
        # TODO: only support 1 rnn layer now
        return StackedRNNDesign(
            name=name,
            data_width=self.inputs_QParams.quant_bits,
            rnn_layers=self.rnn_layers,
            num_layers=self.num_layers,
            work_library_name="work",
        )

    def precompute(self) -> None:
        for layer in self.rnn_layers:
            layer.precompute()

        h_prev = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32).to(
            "cuda"
        )
        c_prev = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32).to(
            "cuda"
        )
        noise = torch.randn_like(h_prev) * 1e-3
        h_prev = h_prev + noise
        c_prev = c_prev + noise

        self.q_h_prev = self.h_prev_QParams.quantize(h_prev).to("cpu")
        self.q_c_prev = self.c_prev_QParams.quantize(c_prev).to("cpu")

        self.precomputed = True

    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")
        q_h_prev = self.q_h_prev
        q_c_prev = self.q_c_prev

        for layer in self.rnn_layers:
            q_outputs, q_h_next, q_c_next = layer.int_forward(
                q_inputs=q_inputs,
                q_h_prev=q_h_prev,
                q_c_prev=q_c_prev,
            )
            q_inputs = q_outputs
            q_h_prev = q_h_next
            q_c_prev = q_c_next

        save_quant_data(q_outputs[:, -1, :], self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs[:, -1, :]

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(inputs)

        self.batch_size = inputs.size(0)

        h_prev = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32).to(
            inputs.device
        )
        c_prev = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32).to(
            inputs.device
        )
        noise = torch.randn_like(h_prev) * 1e-3
        h_prev = h_prev + noise
        c_prev = c_prev + noise

        if self.training:
            self.h_prev_QParams.update_quant_params(h_prev)
            self.c_prev_QParams.update_quant_params(c_prev)

        given_inputs_QParams = self.inputs_QParams
        given_h_prev_QParams = self.h_prev_QParams
        given_c_prev_QParams = self.c_prev_QParams

        for layer in self.rnn_layers:
            outputs, h_next, c_next = layer.forward(
                inputs=inputs,
                h_prev=h_prev,
                c_prev=c_prev,
                given_inputs_QParams=given_inputs_QParams,
                given_h_prev_QParams=given_h_prev_QParams,
                given_c_prev_QParams=given_c_prev_QParams,
            )
            inputs = outputs
            h_prev = h_next
            c_prev = c_next
            given_inputs_QParams = layer.outputs_QParams
            given_h_prev_QParams = layer.h_next_QParams
            given_c_prev_QParams = layer.c_next_QParams

        self.outputs_QParams = self.rnn_layers[-1].h_next_QParams
        return outputs[:, -1, :]
