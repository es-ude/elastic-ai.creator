from elasticai.creator.input_domains import create_codomain_for_1d_conv
from elasticai.creator.vhdl.generator.lut_conv import NaiveLUTBasedConv
from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string
import torch

from elasticai.creator.vhdl.number_representations import ToLogicEncoder


def main(example_inputs, example_outputs, file_path):

    with open(file_path, "w") as writer:
        input_logic_encoder = ToLogicEncoder()
        input_logic_encoder.add_numeric(-1)
        input_logic_encoder.add_numeric(1)
        output_logic_encoder = ToLogicEncoder()
        output_logic_encoder.add_numeric(0)
        output_logic_encoder.add_numeric(1)
        encoded_inputs = []
        encoded_outputs = []
        for input, output in zip(example_inputs, example_outputs):
            encoded_inputs.append(list(map(lambda x: input_logic_encoder(x[0]), input)))
            encoded_outputs.append(list(map(lambda x: output_logic_encoder(x), output)))
        LUT = NaiveLUTBasedConv(
            inputs=encoded_inputs,
            outputs=encoded_outputs,
            input_width=input_logic_encoder.bit_width,
            output_width=output_logic_encoder.bit_width,
        )
        for line in LUT():
            writer.write(line)
            if line[-1] != "\n":
                writer.write("\n")


if __name__ == "__main__":
    file_path = get_file_path_string(
        folder_names=["..", "source"], file_name="lut_conv.vhd"
    )
    example_inputs = create_codomain_for_1d_conv(
        shape=(1, 2), codomain_elements=[-1, 1]
    ).tolist()
    example_outputs = torch.zeros((4, 1))
    example_outputs[[0, 2]] = 1
    example_outputs = example_outputs.tolist()
    main(example_inputs, example_outputs, file_path)
