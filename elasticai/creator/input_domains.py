import itertools
from typing import Dict, Iterable, List, Union

import numpy as np
import torch

from elasticai.creator.mlframework import Module, Tensor


def create_input_data(
    input_dim: List[int], domain: List[Union[int, float]], dtype="float16"
) -> np.ndarray:
    """The function creates possible inputs for a submodel given input dimensions .
    Args:
        input_dim (List[int]): The dimensions of the input, can be a tuple too.
        domain (List[Union[int, float]]): The possible values of the input

    Returns:
        numpy.ndarray: An array of possible inputs
    """
    input_dim = list(input_dim)
    input_dim = [x for x in input_dim if x is not None]
    domain = np.unique(domain)
    n_values = int(len(domain) ** np.prod(input_dim))
    table = np.zeros([n_values] + input_dim, dtype=dtype)

    for vector, i in zip(
        itertools.product(domain, repeat=int(np.prod(input_dim))), range(n_values)
    ):
        table[i] = np.reshape(np.asarray(vector), input_dim)

    return table


def get_cartesian_product_from_items(
    length: int, items: Iterable, dtype="float16"
) -> np.ndarray:
    return np.array(tuple(itertools.product(items, repeat=length)), dtype=dtype)


def construct_codomain_from_elements(
    shape: tuple[int, ...], codomain_elements: Iterable, dtype="float16"
) -> np.ndarray:
    """Build the numpy array containing all combinations that can be built from `items` resulting in the desired `shape`.

    `items` can either be a flat iterable of scalar values or already structure that is convertible to a nested numpy array.
    In the latter case the last dimensions of the requested `shape` are expected to match the shape of `items` exactly.
    Otherwise a `ValueError` will be raised.
    E.g.:
    ```
      construct_domain_from_items((3, 2), ((1, 1), (0, 0))) # is fine
      construct_domain_from_items((2, 3), ((1, 1), (0, 0))) # will raise a ValueError
    ```
    """
    result = np.array(codomain_elements)

    shape_we_need_to_build = shape[
        : _calculate_the_rank_index_that_shape_sizes_match_up_to(
            requested_shape=shape, provided_shape=result.shape[1:]
        )
    ]
    for size in shape_we_need_to_build:
        result = get_cartesian_product_from_items(
            length=size, items=result, dtype=dtype
        )
    return result


def create_io_table(
    inputs: np.ndarray, outputs: List[Tensor], channel_wise=True
) -> Union[Dict, List[Dict]]:
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
        io_table: Union[list[dict], dict] = [{} for _ in range(num_channels)]
        for input, output in zip(inputs, outputs):
            for channel in range(num_channels):
                io_table[channel][tuple(input[:, channel].flatten().tolist())] = tuple(
                    output[:, channel].detach().numpy().flatten().tolist()
                )
    else:
        io_table = {}
        for input, output in zip(inputs, outputs):
            io_table[tuple(input.flatten().tolist())] = tuple(
                output.detach().numpy().flatten().tolist()
            )
    return io_table


def find_unique_elements(input: np.ndarray) -> np.ndarray:
    """Filters an input into its unique elements.
    Args:
        input (np.ndarray): The array to be filtered

    Returns:
        numpy.ndarray: A set array
    """
    output = np.unique(input, axis=0)
    return output


def depthwise_inputs(
    conv_layer: Module,
    kernel_size: tuple[int],
    codomain: List[Union[int, float]],
) -> np.ndarray:
    """Calculate the inputs for depthwise convolution .
    Args:
        conv_layer: Layer, the convolutional layer
        codomain: List[int], the codomain for the block
        outputs: list the outputs from a previous layer
    Returns:
      np.ndarray: The inputs
    """
    n_channels: int = conv_layer.in_channels
    inputs = create_input_data(list(kernel_size), codomain)
    inputs = np.reshape(inputs, [-1, 1, *kernel_size])
    stacked_inputs = inputs.copy()

    if n_channels > 1:
        for _ in range(n_channels - 1):
            stacked_inputs = np.concatenate([stacked_inputs, inputs], axis=-2)

    return stacked_inputs


def _calculate_the_rank_index_that_shape_sizes_match_up_to(
    requested_shape, provided_shape
):
    rank_size_pairs = tuple(zip(reversed(provided_shape), reversed(requested_shape)))
    equalities = map(lambda x: x[0] == x[1], rank_size_pairs)
    if not all(equalities):
        raise ValueError
    return len(requested_shape) - len(rank_size_pairs)


def create_codomain_for_1d_conv(shape, codomain_elements):
    domain: torch.Tensor = torch.as_tensor(
        construct_codomain_from_elements(
            shape=shape, codomain_elements=codomain_elements, dtype="float32"
        )
    )
    return domain


def create_codomain_for_depthwise_1d_conv(shape, codomain_elements) -> Tensor:
    kernel_size = shape[1]
    channels = shape[0]
    domain: torch.Tensor = torch.as_tensor(
        construct_codomain_from_elements(
            shape=(kernel_size,), codomain_elements=codomain_elements, dtype="float32"
        )
    )
    domain = domain.reshape([-1, 1, kernel_size])
    domain = domain.repeat_interleave(channels, dim=1)
    return domain
