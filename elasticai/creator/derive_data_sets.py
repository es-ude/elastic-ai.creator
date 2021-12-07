from typing import List, Dict, Union
import numpy as np
import itertools as ite
import sys
from torch import Tensor
from torch.nn import Module


def create_input_data(input_dim: List[int], domain: List[int]) -> np.ndarray:
    """The function creates possible inputs for a submodel given input dimensions .
        Args:
            input_dim (List[int]): The dimensions of the input, can be a tuple too.
            domain (List[int]): The possible values of the input

        Returns:
            numpy.ndarray: An array of possible inputs
        """
    input_dim = list(input_dim)
    input_dim = [x for x in input_dim if x is not None]
    domain = np.unique(domain)
    n_values = int(len(domain) ** np.prod(input_dim).item())
    try:
        if hasattr(domain, "shape") and len(domain.shape) > 1:
            table = np.zeros([n_values] + input_dim + list(domain.shape[1:]))
        else:
            table = np.zeros([n_values] + input_dim)
        for vector, i in zip(ite.product(domain, repeat=int(np.prod(input_dim))), range(n_values)):
            if hasattr(domain, "shape") and len(domain.shape) > 1:
                table[i] = np.reshape(np.asarray(vector), input_dim + list(domain.shape[1:]))
            else:
                table[i] = np.reshape(np.asarray(vector), input_dim)
    except MemoryError:
        print(
            "MemoryError: The generated tables are too large. No convertion possible. Unable to allocate memory for an array with shape ({}, {})".format(
                n_values, input_dim
            ))
        sys.exit(1)
    return table


def create_io_table(inputs: np.ndarray, outputs: List[Tensor], channel_wise=True) -> Union[Dict, List[Dict]]:
    """The function creates io tables given an array of inputs and a model .
            Args:
                inputs (np.ndarray): The inputs
                outputs (List [torch.Tensor]): The outputs
                channel_wise (bool): default is False. If True, function returns a list with one table per channel in inputs.

            Returns:
                Dict: The inputs mapped to outputs
            """
    if channel_wise:
        num_channels = inputs.shape[-1]
        io_table = [{} for _ in range(num_channels)]
        for input, output in zip(inputs, outputs):
            for channel in range(num_channels):
                io_table[channel][tuple(input[:, channel].flatten().tolist())] = tuple(
                    output[:, channel].detach().numpy().flatten().tolist()
                )
    else:
        io_table = {}
        for input, output in zip(inputs, outputs):
            io_table[tuple(input.flatten().tolist())] = tuple(output.detach().numpy().flatten().tolist())
    return io_table


def find_unique_elements(input: np.ndarray) -> np.ndarray:
    """Filters an input into its unique elements.
            Args:
                input (np.ndarray): The array to be filtered

            Returns:
                numpy.ndarray: A set array
            """
    original_shape = list(input.shape)
    output = np.unique(input, axis=0)

    return output.reshape(original_shape[1:].insert(0, output.shape[0]))


def depthwise_inputs(conv_layer: Module,
                     kernel_size: tuple[int],
                     codomain: List[int],
                     ) -> np.ndarray:
    """Calculate the inputs for depthwise convolution .
        Args:
            conv_layer: Layer, the convolutional layer
            codomain: List[int], the codomain for the block
            outputs: list the outputs from a previous layer
        Returns:
          np.ndarray: The inputs
    """
    n_channels = conv_layer.in_channels
    inputs = create_input_data(list(kernel_size), codomain)
    inputs = np.reshape(inputs, [-1, 1, *kernel_size])
    stacked_inputs = inputs.copy()

    if n_channels > 1:
        for _ in range(n_channels - 1):
            stacked_inputs = np.concatenate([stacked_inputs, inputs], axis=-2)

    return stacked_inputs
