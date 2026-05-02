import random
from typing import Protocol

import elasticai.creator.ir as ir
from elasticai.creator.ir import attribute
from elasticai.creator.ir2vhdl import Shape

from .rules._ir import DataGraph, Node
from .rules._ir import ir_factory as factory


class Caller(Protocol):
    def __call__(self, *inputs: Node) -> Node: ...


class NetworkBuilder:
    def __init__(self):
        self._reg = ir.Registry(
            {
                "root": factory.graph().add_nodes(
                    ("input", ir.attribute(type="input")),
                    ("output", ir.attribute(type="output")),
                )
            }
        )
        self._names = ir.NameRegistry()
        self._root_nodes = ir.NameRegistry()

    @property
    def _root(self) -> DataGraph:
        return self._reg["root"]

    @_root.setter
    def _root(self, val: DataGraph) -> None:
        self._reg = self._reg.add("root", val)

    def add(self, layer: DataGraph) -> Caller:
        _type = layer.type
        name = self._names.get_unique_name(_type)
        self._reg = self._reg.add(name, layer)

        def caller(*inputs: Node) -> Node:
            root = self._reg["root"]
            edges: list[tuple[str, str]] = []
            call_node = factory.node(
                self._root_nodes.get_unique_name(_type),
                attribute(type=_type, implementation=name),
            )

            for n in inputs:
                edges.append((n.name, call_node.name))
            root = root.add_edges(*edges).add_node(call_node)
            self._reg = self._reg.add("root", root)
            return call_node

        return caller

    def open(self, input_shape: Shape) -> Node:
        self._root = self._root.add_node(
            "input",
            attribute(
                type="input",
                output_shape=input_shape.to_tuple(),
                input_shape=input_shape.to_tuple(),
            ),
        )
        return self._reg["root"].nodes["input"]

    def close(self, *inputs: Node) -> ir.Registry[DataGraph]:
        reg = self._reg.add(
            "root", self._reg["root"].add_edges(*((n.name, "output") for n in inputs))
        )
        self._reg = ir.Registry()
        self._names = ir.NameRegistry()
        self._root_nodes = ir.NameRegistry()
        return reg


def linear(
    in_features: int,
    out_features: int,
    weight: list[list[float]] | None = None,
    bias: list[float] | None = None,
    use_bias: bool = True,
) -> DataGraph:

    if weight is None:
        weight = []
        for _ in range(out_features):
            out_feature = []
            for _ in range(in_features):
                out_feature.append(random.gauss())
            weight.append(out_feature)
    if use_bias and bias is None:
        bias = [random.gauss() for _ in range(out_features)]

    parameters = (
        attribute(weight=weight)
        if bias is None
        else attribute(weight=weight, bias=bias)
    )
    return factory.graph(
        attribute(
            type="linear",
            in_features=in_features,
            out_features=out_features,
            bias=use_bias,
            parameters=parameters,
        )
    )


def conv1d(
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    groups: int = 1,
    stride: int = 1,
    weight: list[list[list[float]]] | None = None,
    bias: list[float] | None = None,
    use_bias: bool = True,
    input_bitwidth: int = 1,
) -> DataGraph:
    if weight is None:
        weight = []
        for _ in range(out_channels):
            outs = []
            for _ in range(in_channels):
                ins = []
                for _ in range(kernel_size):
                    ins.append(random.gauss())
                outs.append(ins)
            weight.append(outs)
    if bias is None and use_bias:
        bias = [random.gauss() for _ in range(out_channels)]
    parameters = (
        attribute(weight=weight)
        if bias is None
        else attribute(weight=weight, bias=bias)
    )
    return factory.graph(
        attribute(
            type="conv1d",
            groups=groups,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=use_bias,
            kernel_size=kernel_size,
            input_bitwidth=input_bitwidth,
            parameters=parameters,
        )
    )


def maxpool(kernel_size: int, stride: int) -> DataGraph:
    return factory.graph(
        attribute(type="maxpool1d", kernel_size=kernel_size, stride=stride)
    )


def batchnorm1d(
    num_features: int,
    affine: bool = False,
    weight: list[float] | None = None,
    bias: list[float] | None = None,
    running_mean: list[float] | None = None,
    running_var: list[float] | None = None,
) -> DataGraph:
    if weight is None and affine:
        weight = [1.0 for _ in range(num_features)]
    if bias is None and affine:
        bias = [0.0 for _ in range(num_features)]
    if running_mean is None:
        running_mean = [0.0 for _ in range(num_features)]
    if running_var is None:
        running_var = [1.0 for _ in range(num_features)]
    args = {}
    if affine:
        args["weight"] = weight
        args["bias"] = bias

    return factory.graph(
        attribute(
            type="batchnorm1d",
            num_features=num_features,
            affine=affine,
            running_mean=running_mean,
            running_var=running_var,
        )
        | attribute(parameters=args)
    )


def binarize() -> DataGraph:
    return factory.graph(attribute(type="binarize"))
