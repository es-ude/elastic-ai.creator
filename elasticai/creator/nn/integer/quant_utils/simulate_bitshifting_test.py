import torch

from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)


def test_simulate_bitshifting():
    # float32
    input = torch.tensor([[1.5000, -2.8000], [-9.0000, 4.2000]], dtype=torch.float32)
    weight = torch.tensor([[-1.0000, -0.3800], [0.5000, 1.0000]], dtype=torch.float32)
    bias = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    output = torch.tensor([[-1.4450, -1.0453], [6.3950, 0.7071]], dtype=torch.float32)

    # int8
    input_scale = torch.tensor([0.0518], dtype=torch.float32)
    input_zero_point = torch.tensor([46], dtype=torch.int32)
    q_input = torch.tensor([[75, -8], [-128, 127]], dtype=torch.int32)

    weight_scale = torch.tensor([0.0078], dtype=torch.float32)
    weight_zero_point = torch.tensor([-1], dtype=torch.int32)
    q_weight = torch.tensor([[-128, -49], [63, 126]], dtype=torch.int32)

    bias_scale = input_scale * weight_scale
    bias_zero_point = torch.tensor([0], dtype=torch.int32)
    q_bias = torch.tensor([-2463, 2463], dtype=torch.int32)

    output_scale = torch.tensor([0.0307], dtype=torch.float32)
    output_zero_point = torch.tensor([-81], dtype=torch.int32)
    q_output = torch.tensor([[-128, -115], [127, -58]], dtype=torch.int32)

    scale_factor_M = input_scale * weight_scale / output_scale
    m_q_shift = torch.tensor([21], dtype=torch.int32)
    m_q = torch.tensor([27703], dtype=torch.int32)

    tmp = (q_input - input_zero_point).mm(q_weight.t() - weight_zero_point) + q_bias
    result = simulate_bitshifting(tmp, m_q_shift, m_q)

    expected_result = torch.tensor([[-47, -34], [208, 21]], dtype=torch.int32)
    torch.testing.assert_close(result, expected_result)
