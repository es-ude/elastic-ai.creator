from typing import Any

import torch
import torch.fx as fx

from elasticai.creator.qat.layers import Binarize


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.test_lstm = torch.nn.LSTM(input_size=1, hidden_size=10)
        self.batch_norm = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.test_lstm(x)


class LSTMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm_container = MyModule()
        self.linear = torch.nn.Linear(in_features=10, out_features=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(self.lstm_container(x)[0])


if __name__ == "__main__":
    model = LSTMModel()

    tracer = fx.Tracer()
    graph = tracer.trace(model)

    new_graph = fx.Graph()
    node_args: dict[str, fx.Node] = {}

    def transform_argument_for_old_node_to_new_node(node_argument: fx.Node):
        return node_args[node_argument.name]

    gm: fx.GraphModule = fx.GraphModule(model, graph)

    def prepare_arguments_for_new_node(arg: Any) -> Any:
        if isinstance(arg, fx.Node):
            return node_args[arg.name]
        return arg

    for node in graph.nodes:
        if node.op == "output":
            new_graph.output(node_args["output"])
        else:
            if node.op == "call_module" and node.target == "linear":
                model.binarize = Binarize()
                new_node = new_graph.call_module(
                    module_name="binarize",
                    args=tuple(
                        [prepare_arguments_for_new_node(arg) for arg in node.args]
                    ),
                )
                node_args[node.name] = new_node
                new_node = new_graph.call_module(
                    "lstm_container.batch_norm", args=(new_node,)
                )
                node_args["output"] = new_node
            else:
                new_node = new_graph.node_copy(
                    node, transform_argument_for_old_node_to_new_node
                )
                node_args[node.name] = new_node
    print(new_graph)

    gm = fx.GraphModule(model, new_graph)

    print(gm)
    print(gm.graph)
    gm.graph.lint()

    print(tracer.trace(gm))
    gm(torch.tensor([[[1.0], [1.0]]]))
