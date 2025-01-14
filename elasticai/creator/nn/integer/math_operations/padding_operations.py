import torch
import torch.nn.functional as F


def get_padding_len(
    padding: tuple[int, int] or str,
    kernel_size: tuple[int, int] or int,
) -> int:
    kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size

    if isinstance(padding, tuple):
        padding_len = padding[0]
    elif isinstance(padding, str) and padding == "same":  # TODO:support "valid" padding
        padding_len = kernel_size // 2
    elif isinstance(padding, int):
        padding_len = padding
    else:
        raise ValueError(f"Unsupported padding type or value: {padding}")
    return padding_len


def get_padded_q_inputs(
    padding_len: int,
    q_inputs: torch.IntTensor,
    inputs_QParams,
) -> None:
    if padding_len > 0:
        return F.pad(
            input=q_inputs,
            pad=(padding_len, padding_len),
            mode="constant",
            value=inputs_QParams.zero_point.item(),
        )
    return q_inputs


def get_padded_count(
    padding: tuple[int, int] or str,
    kernel_size: tuple[int, int] or int,
    in_channels: int,
    out_channels: int,
    seq_len: int,
) -> int:
    x_count = in_channels * seq_len

    kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    if isinstance(padding, (tuple, int)):
        padding_len = padding[0] if isinstance(padding, tuple) else padding
        if padding_len == 0:
            y_count = out_channels * (seq_len - kernel_size + 1)
        else:
            y_count = out_channels * (seq_len + 2 * padding_len - kernel_size + 1)
    elif isinstance(padding, str):
        if padding == "same":
            y_count = out_channels * seq_len
        else:
            raise ValueError(f"Unsupported padding value: {padding}")
    else:
        raise ValueError(f"Unsupported padding type: {type(padding)}")

    return x_count, y_count


def get_vhdl_templates(padding_len, layer_name):
    if padding_len == 0:
        suffix = "_not_padding"
    else:
        suffix = "_zero_padding"

    template_file_name = f"{layer_name}{suffix}.tpl.vhd"
    test_template_file_name = f"{layer_name}{suffix}_tb.tpl.vhd"

    return template_file_name, test_template_file_name
