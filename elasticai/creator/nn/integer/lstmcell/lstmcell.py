import torch
import torch.nn as nn

from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.concatenate import Concatenate
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.hadamardproduct import HadamardProduct
from elasticai.creator.nn.integer.hardsigmoid import HardSigmoid
from elasticai.creator.nn.integer.hardtanh import HardTanh
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.lstmcell.design import LSTMCell as LSTMCellDesign
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class LSTMCell(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        inputs_size = kwargs.get("inputs_size")
        hidden_size = kwargs.get("hidden_size")
        window_size = kwargs.get("window_size")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.concatenate = Concatenate(
            name=self.name + "_concatenate",
            num_features=inputs_size + hidden_size,
            num_dimensions=1,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.f_gate_linear = Linear(
            name=self.name + "_f_gate_linear",
            in_features=inputs_size + hidden_size,
            out_features=hidden_size,
            num_dimensions=window_size,
            bias=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.c_gate_linear = Linear(
            name=self.name + "_c_gate_linear",
            in_features=inputs_size + hidden_size,
            out_features=hidden_size,
            num_dimensions=window_size,
            bias=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.i_gate_linear = Linear(
            name=self.name + "_i_gate_linear",
            in_features=inputs_size + hidden_size,
            out_features=hidden_size,
            bias=True,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.o_gate_linear = Linear(
            name=self.name + "_o_gate_linear",
            in_features=inputs_size + hidden_size,
            out_features=hidden_size,
            bias=True,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.i_sigmoid = HardSigmoid(
            name=self.name + "_i_sigmoid",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.f_sigmoid = HardSigmoid(
            name=self.name + "_f_sigmoid",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.o_sigmoid = HardSigmoid(
            name=self.name + "_o_sigmoid",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.c_tanh = HardTanh(
            name=self.name + "_c_tanh",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.c_next_tanh = HardTanh(
            name=self.name + "_c_next_tanh",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.c_next_addition = Addition(
            name=self.name + "_add",
            num_features=window_size,
            num_dimensions=hidden_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.fc_hadamard_product = HadamardProduct(
            name=self.name + "_fc_hadamard_product",
            num_features=hidden_size,  # TODO: check this
            num_dimensions=1,  # TODO: check this
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.ic_hadamard_product = HadamardProduct(
            name=self.name + "_ic_hadamard_product",
            num_features=hidden_size,
            num_dimensions=1,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.oc_hadamard_product = HadamardProduct(
            name=self.name + "_oc_hadamard_product",
            num_features=hidden_size,
            num_dimensions=1,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_prev_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_next_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.c_prev_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.c_next_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name) -> LSTMCellDesign:
        return LSTMCellDesign(
            name=name,
            data_width=self.quant_bits,
            concatenate=self.concatenate,
            f_gate_linear=self.f_gate_linear,
            c_gate_linear=self.c_gate_linear,
            i_gate_linear=self.i_gate_linear,
            o_gate_linear=self.o_gate_linear,
            i_sigmoid=self.i_sigmoid,
            f_sigmoid=self.f_sigmoid,
            o_sigmoid=self.o_sigmoid,
            c_tanh=self.c_tanh,
            c_next_tanh=self.c_next_tanh,
            c_next_addition=self.c_next_addition,
            fc_hadamard_product=self.fc_hadamard_product,
            ic_hadamard_product=self.ic_hadamard_product,
            oc_hadamard_product=self.oc_hadamard_product,
            work_library_name="work",
        )

    def precompute(self) -> None:
        self.concatenate.precompute()
        self.i_gate_linear.precompute()
        self.f_gate_linear.precompute()
        self.c_gate_linear.precompute()
        self.o_gate_linear.precompute()

        self.i_sigmoid.precompute()
        self.f_sigmoid.precompute()
        self.o_sigmoid.precompute()

        self.c_tanh.precompute()
        self.c_next_tanh.precompute()

        self.c_next_addition.precompute()

        self.fc_hadamard_product.precompute()
        self.ic_hadamard_product.precompute()
        self.oc_hadamard_product.precompute()

        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
        q_h_prev: torch.IntTensor,
        q_c_prev: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x_1")
        save_quant_data(q_h_prev, self.quant_data_dir, f"{self.name}_q_x_2")
        save_quant_data(q_c_prev, self.quant_data_dir, f"{self.name}_q_x_3")

        # concatenate inputs and h_prev
        q_concated_inputs = self.concatenate.int_forward(
            q_inputs1=q_inputs, q_inputs2=q_h_prev
        )

        # gate linear transformations
        q_i_gate_outputs = self.i_gate_linear.int_forward(q_inputs=q_concated_inputs)
        q_f_gate_outputs = self.f_gate_linear.int_forward(q_inputs=q_concated_inputs)
        q_o_gate_outputs = self.o_gate_linear.int_forward(q_inputs=q_concated_inputs)
        q_c_gate_outputs = self.c_gate_linear.int_forward(q_inputs=q_concated_inputs)

        # gate activations
        q_i_gate_sigmoid_outputs = self.i_sigmoid.int_forward(q_inputs=q_i_gate_outputs)
        q_f_gate_sigmoid_outputs = self.f_sigmoid.int_forward(q_inputs=q_f_gate_outputs)
        q_o_gate_sigmoid_outputs = self.o_sigmoid.int_forward(q_inputs=q_o_gate_outputs)
        q_c_gate_tanh_outputs = self.c_tanh.int_forward(q_inputs=q_c_gate_outputs)

        # next cell state
        q_c_next_inputs1 = self.fc_hadamard_product.int_forward(
            q_inputs1=q_f_gate_sigmoid_outputs, q_inputs2=q_c_prev
        )
        q_c_next_inputs2 = self.ic_hadamard_product.int_forward(
            q_inputs1=q_i_gate_sigmoid_outputs, q_inputs2=q_c_gate_tanh_outputs
        )
        q_c_next = self.c_next_addition.int_forward(
            q_inputs1=q_c_next_inputs1, q_inputs2=q_c_next_inputs2
        )

        # next hidden state
        q_c_next_tanh_ouputs = self.c_next_tanh.int_forward(q_inputs=q_c_next)
        q_h_next = self.oc_hadamard_product.int_forward(
            q_inputs1=q_o_gate_sigmoid_outputs, q_inputs2=q_c_next_tanh_ouputs
        )

        save_quant_data(q_h_next, self.quant_data_dir, f"{self.name}_q_y_1")
        save_quant_data(q_c_next, self.quant_data_dir, f"{self.name}_q_y_2")

        return q_h_next, q_c_next

    def forward(
        self,
        inputs: torch.FloatTensor,
        h_prev: torch.FloatTensor,
        c_prev: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module,
    ) -> torch.Tensor:
        self.inputs_QParams = given_inputs_QParams

        if self.training:
            self.h_prev_QParams.update_quant_params(h_prev)
            self.c_prev_QParams.update_quant_params(c_prev)

        # concatenate inputs and h_prev
        concated_inputs = self.concatenate.forward(
            inputs1=inputs,
            inputs2=h_prev,
            given_inputs1_QParams=self.inputs_QParams,
            given_inputs2_QParams=self.h_prev_QParams,
        )

        # gate linear transformations
        i_gate_outputs = self.i_gate_linear.forward(
            inputs=concated_inputs,
            given_inputs_QParams=self.concatenate.outputs_QParams,
        )
        f_gate_outputs = self.f_gate_linear.forward(
            inputs=concated_inputs,
            given_inputs_QParams=self.concatenate.outputs_QParams,
        )
        o_gate_outputs = self.o_gate_linear.forward(
            inputs=concated_inputs,
            given_inputs_QParams=self.concatenate.outputs_QParams,
        )
        c_gate_outputs = self.c_gate_linear.forward(
            inputs=concated_inputs,
            given_inputs_QParams=self.concatenate.outputs_QParams,
        )

        # gate activations
        i_gate_sigmoid_outputs = self.i_sigmoid.forward(
            inputs=i_gate_outputs,
            given_inputs_QParams=self.i_gate_linear.outputs_QParams,
        )
        f_gate_sigmoid_outputs = self.f_sigmoid.forward(
            inputs=f_gate_outputs,
            given_inputs_QParams=self.f_gate_linear.outputs_QParams,
        )
        o_gate_sigmoid_outputs = self.o_sigmoid.forward(
            inputs=o_gate_outputs,
            given_inputs_QParams=self.o_gate_linear.outputs_QParams,
        )
        c_gate_tanh_outputs = self.c_tanh.forward(
            inputs=c_gate_outputs,
            given_inputs_QParams=self.c_gate_linear.outputs_QParams,
        )

        # next cell state
        c_next_inputs1 = self.fc_hadamard_product.forward(
            inputs1=f_gate_sigmoid_outputs,
            inputs2=c_prev,
            given_inputs1_QParams=self.f_sigmoid.outputs_QParams,
            given_inputs2_QParams=self.c_prev_QParams,
        )

        c_next_inputs2 = self.ic_hadamard_product.forward(
            inputs1=i_gate_sigmoid_outputs,
            inputs2=c_gate_tanh_outputs,
            given_inputs1_QParams=self.i_sigmoid.outputs_QParams,
            given_inputs2_QParams=self.c_tanh.outputs_QParams,
        )
        c_next = self.c_next_addition.forward(
            inputs1=c_next_inputs1,
            inputs2=c_next_inputs2,
            given_inputs1_QParams=self.fc_hadamard_product.outputs_QParams,
            given_inputs2_QParams=self.ic_hadamard_product.outputs_QParams,
        )

        # next hidden state
        c_next_tanh_outputs = self.c_next_tanh.forward(
            inputs=c_next, given_inputs_QParams=self.c_next_addition.outputs_QParams
        )
        h_next = self.oc_hadamard_product.forward(
            inputs1=o_gate_sigmoid_outputs,
            inputs2=c_next_tanh_outputs,
            given_inputs1_QParams=self.o_sigmoid.outputs_QParams,
            given_inputs2_QParams=self.c_next_tanh.outputs_QParams,
        )
        self.h_next_QParams = self.oc_hadamard_product.outputs_QParams
        self.c_next_QParams = self.c_next_addition.outputs_QParams

        return h_next, c_next
