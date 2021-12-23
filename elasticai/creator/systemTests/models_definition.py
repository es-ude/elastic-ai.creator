import torch
from torch import nn

import brevitas.nn as bnn
from torch.nn.utils.parametrize import is_parametrized

from elasticai.creator.layers import Binarize, QConv1d, QLinear
import elasticai.creator.brevitas.brevitas_quantizers as bquant


def create_qtorch_model():
    model = nn.Sequential(
        QConv1d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            quantizer=Binarize(),
            bias=False,
        ),
        nn.BatchNorm1d(num_features=3),
        Binarize(),
        nn.MaxPool1d(
            kernel_size=5,
            stride=3,
        ),
        QConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            quantizer=Binarize(),
        ),
        nn.BatchNorm1d(num_features=3),
        Binarize(),
        QConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            quantizer=Binarize(),
        ),
        nn.BatchNorm1d(num_features=3),
        Binarize(),
        nn.MaxPool1d(
            kernel_size=3,
            stride=2,
        ),
        QConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=2,
            quantizer=Binarize(),
        ),
        nn.BatchNorm1d(num_features=3),
        Binarize(),
        QConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=2,
            quantizer=Binarize(),
        ),
        nn.BatchNorm1d(num_features=3),
        Binarize(),
        nn.MaxPool1d(
            kernel_size=3,
            stride=2,
        ),
        nn.Flatten(),
        QLinear(in_features=15, out_features=1, quantizer=Binarize(), bias=False),
        nn.Sigmoid(),
    )
    return model


def create_brevitas_model():
    model = nn.Sequential(
        bnn.QuantConv1d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            bias=False,
            weight_quant=bquant.BinaryWeights,
        ),
        nn.BatchNorm1d(num_features=3),
        bnn.QuantIdentity(act_quant=bquant.BinaryActivation),
        nn.MaxPool1d(kernel_size=5, stride=3),
        bnn.QuantConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
        ),
        nn.BatchNorm1d(num_features=3),
        bnn.QuantIdentity(act_quant=bquant.BinaryActivation),
        bnn.QuantConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
        ),
        nn.BatchNorm1d(num_features=3),
        bnn.QuantIdentity(act_quant=bquant.BinaryActivation),
        nn.MaxPool1d(kernel_size=3, stride=2),
        bnn.QuantConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=2,
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
        ),
        nn.BatchNorm1d(num_features=3),
        bnn.QuantIdentity(act_quant=bquant.BinaryActivation),
        bnn.QuantConv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=2,
            weight_quant=bquant.BinaryWeights,
            bias_quant=bquant.BinaryBias,
        ),
        nn.BatchNorm1d(num_features=3),
        bnn.QuantIdentity(act_quant=bquant.BinaryActivation),
        nn.MaxPool1d(
            kernel_size=3,
            stride=2,
        ),
        nn.Flatten(),
        bnn.QuantLinear(
            in_features=15,
            out_features=1,
            bias=False,
            weight_quant=bquant.BinaryWeights,
        ),
        nn.Sigmoid(),
    )
    return model


weight_layers = (QConv1d, QLinear, nn.BatchNorm1d, bnn.QuantConv1d, bnn.QuantLinear)


def define_weight(layers):
    # TODO: In case we're using this function just to have deterministic weight value generation
    #  we should reconsider our approach here
    for layer in layers:
        if isinstance(layer, weight_layers):
            if is_parametrized(layer):
                layer.weight = torch.ones_like(layer.weight)
            else:
                layer.weight = torch.nn.Parameter(torch.ones_like(layer.weight))
            if layer.bias is not None:
                if is_parametrized(layer):
                    layer.bias = torch.ones_like(layer.bias)
                else:
                    layer.bias = torch.nn.Parameter(torch.ones_like(layer.bias))
