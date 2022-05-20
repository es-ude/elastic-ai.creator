import brevitas.nn as bnn
from torch import nn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.layers import Binarize, QConv1d, QLinear


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
