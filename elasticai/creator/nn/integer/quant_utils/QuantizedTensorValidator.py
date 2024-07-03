import torch


class QuantizedTensorValidator:
    @staticmethod
    def check_dtype(tensor, tensor_name, expected_dtype, logger):
        if tensor.dtype != expected_dtype:
            logger.error(
                f"{tensor_name} dtype mismatch: expected {expected_dtype}, got"
                f" {tensor.dtype}"
            )
            raise TypeError(f"{tensor_name} must be a {expected_dtype}")

    @staticmethod
    def check_drange(tensor, tensor_name, lower_bound, upper_bound, logger):
        if torch.any(tensor < lower_bound) or torch.any(tensor > upper_bound):
            logger.warning(f"{tensor_name} out of range [{lower_bound}, {upper_bound}]")
