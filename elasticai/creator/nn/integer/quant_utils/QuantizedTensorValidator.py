import logging

import torch


class QuantizedTensorValidator:
    @staticmethod
    def check_dtype(
        tensor: torch.Tensor,
        tensor_name: str,
        expected_dtype: torch.dtype,
        logger: logging.Logger,
    ):
        if tensor.dtype != expected_dtype:
            logger.error(
                f"{tensor_name} dtype mismatch: expected {expected_dtype}, got"
                f" {tensor.dtype}"
            )
            raise TypeError(f"{tensor_name} must be a {expected_dtype}")

    @staticmethod
    def check_drange(
        tensor: torch.Tensor,
        tensor_name: str,
        lower_bound: int,
        upper_bound: int,
        logger: logging.Logger,
    ):
        if torch.any(tensor < lower_bound) or torch.any(tensor > upper_bound):
            logger.error(f"{tensor_name} out of range [{lower_bound}, {upper_bound}]")
            raise ValueError(
                f"{tensor_name} values must be within [{lower_bound}, {upper_bound}]"
            )
