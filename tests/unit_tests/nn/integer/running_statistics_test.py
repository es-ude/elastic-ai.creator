"""Running-statistics updates must not retain the training autograd graph."""
import copy
from importlib import import_module

import pytest
import torch
from torch.nn import functional as F

from elasticai.creator.nn.integer.batchnorm1d.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.MPQ.batchnorm1d.batchnorm1d import (
    BatchNorm1d as MPQBatchNorm1d,
)


@pytest.mark.parametrize(
    'module_path',
    [
        'elasticai.creator.nn.integer.pointconv1dbn.pointconv1dbn',
        'elasticai.creator.nn.integer.sepconv1dbn.pointconv1dbn.pointconv1dbn',
    ],
)
@pytest.mark.parametrize('enable_simquant', [False, True])
@pytest.mark.parametrize('batch_size, length', [(4, 8), (1, 2)])
def test_pointconv_running_statistics_do_not_retain_training_graph(
    module_path, enable_simquant, batch_size, length
):
    torch.manual_seed(42)
    module = import_module(module_path)
    layer = module.PointConv1dBN(
        in_channels=2, out_channels=3, seq_len=length,
        name='bn', quant_bits=8, device='cpu',
    )
    _check_running_statistics(
        layer, layer.bn1d, (batch_size, 2, length), enable_simquant, folded=True
    )


@pytest.mark.parametrize('enable_simquant', [False, True])
@pytest.mark.parametrize('batch_size, length', [(4, 8), (1, 2)])
def test_batchnorm_running_statistics_do_not_retain_training_graph(
    enable_simquant, batch_size, length
):
    torch.manual_seed(42)
    layer = BatchNorm1d(
        name='bn',
        norm_dim=3,
        in_features=3,
        out_features=3,
        num_dimensions=length,
        quant_bits=8,
    )
    _check_running_statistics(layer, layer, (batch_size, length, 3), enable_simquant)


@pytest.mark.parametrize('enable_simquant', [False, True])
@pytest.mark.parametrize('batch_size, length', [(4, 8), (1, 2)])
def test_mpq_batchnorm_running_statistics_do_not_retain_training_graph(
    enable_simquant, batch_size, length
):
    torch.manual_seed(42)
    layer = MPQBatchNorm1d(
        name='bn',
        norm_dim=3,
        in_features=3,
        out_features=3,
        num_dimensions=length,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        MPQ_strategy='inheritance',
    )
    layer.set_quant_bits_from_config(
        {f'bn.{key}': 8 for key in ('inputs', 'weights', 'outputs')}
    )
    _check_running_statistics(layer, layer, (batch_size, length, 3), enable_simquant)


def _check_running_statistics(layer, bn, shape, enable_simquant, *, folded=False):
    if not enable_simquant:
        layer.double()  # Resolve near-zero gradient comparisons without float32 cancellation.
    layer.train()
    for _ in range(2):
        layer.zero_grad(set_to_none=True)
        inputs = torch.randn(shape, dtype=next(layer.parameters()).dtype, requires_grad=True)
        old_mean, old_var = bn.running_mean.clone(), bn.running_var.clone()
        outputs = layer(inputs, given_inputs_QParams=None, enable_simquant=enable_simquant)

        assert bn.running_mean.grad_fn is None and not bn.running_mean.requires_grad
        assert bn.running_var.grad_fn is None and not bn.running_var.requires_grad

        if not enable_simquant:
            # Native PyTorch BN uses biased variance for training normalization
            # and unbiased variance for the running buffer. Compare both values
            # and gradients, including batches with only two values per channel.
            if folded:
                values = F.conv1d(inputs, layer.conv1d.weight, layer.conv1d.bias)
            else:
                # These standalone layers use [batch, length, channels] layout.
                values = inputs.transpose(1, 2)
            expected = F.batch_norm(
                values, old_mean, old_var, bn.weight, bn.bias,
                training=True, momentum=bn.momentum, eps=bn.eps,
            )
            if not folded:
                expected = expected.transpose(1, 2)
            torch.testing.assert_close(outputs, expected)
            torch.testing.assert_close(bn.running_mean, old_mean)
            torch.testing.assert_close(bn.running_var, old_var)
            variables = (inputs, *layer.parameters())
            actual_grads = torch.autograd.grad(outputs.square().mean(), variables, retain_graph=True)
            expected_grads = torch.autograd.grad(expected.square().mean(), variables)
            for actual, reference in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-4)
        else:
            accumulator_weight = (
                layer.weight_QParams if folded else layer.modified_weight_QParams
            )
            assert layer.bias_QParams.quant_bits == (19 if folded else 18)
            torch.testing.assert_close(
                layer.bias_QParams.scale_factor,
                layer.inputs_QParams.scale_factor * accumulator_weight.scale_factor,
            )
            outputs.square().mean().backward()
            assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        copy.deepcopy(layer)
