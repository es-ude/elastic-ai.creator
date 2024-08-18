import torch


def calculate_quant_params(
    is_symmetric: bool,
    min_quant: torch.IntTensor,
    max_quant: torch.IntTensor,
    min_float: torch.FloatTensor,
    max_float: torch.FloatTensor,
    eps: torch.FloatTensor,
):
    def _symmetricQuant(min_float: torch.FloatTensor, max_float: torch.FloatTensor):
        # Assuming that min_float and max_float are based on 0 symmetry and are close in absolute value
        max_extent = torch.max(torch.abs(min_float), torch.abs(max_float))
        max_float = max_extent
        min_float = -max_extent

        scale_factor = torch.max(
            ((max_float - min_float) / (max_quant.float() - min_quant.float())), eps
        )
        zero_point = torch.zeros(scale_factor.size())

        return scale_factor, zero_point, min_float, max_float

    def _asymmetricQuant(min_float: torch.FloatTensor, max_float: torch.FloatTensor):
        # Assuming that min_float and max_float are NOT based on 0 symmetry and are NOT close in absolute value
        scale_factor = torch.max(
            ((max_float - min_float) / (max_quant.float() - min_quant.float())), eps
        )
        zero_point = max_quant - (max_float / scale_factor)
        zero_point = zero_point.round_().clamp(min_quant, max_quant)
        zero_point = zero_point.to(min_quant.device)

        ## TODO: check if this is necessary
        # if zero_point < min_quant:
        #     zero_point = torch.tensor([min_quant], dtype=torch.int32).to(min_quant.device)
        # elif zero_point > max_quant:
        #     zero_point = torch.tensor([max_quant], dtype=torch.int32).to(max_quant.device)
        # else:
        #     zero_point = zero_point.to(min_quant.device)

        return scale_factor, zero_point, min_float, max_float

    if is_symmetric:
        return _symmetricQuant(min_float, max_float)
    else:
        return _asymmetricQuant(min_float, max_float)
