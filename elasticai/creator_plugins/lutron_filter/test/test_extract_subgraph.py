from torch.fx import GraphModule, Tracer
from torch.nn import Conv1d, Module, ReLU

from elasticai.creator_plugins.lutron_filter.torch.transformation_utils import (
    move_to_submodule,
)


class Model(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.relu = ReLU()
        self.conv2 = Conv1d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv(x + 1) * 2.0) + 1)


def test_create_correct_submodule():
    m = Model()
    g = Tracer().trace(m)
    name = "new_sub"
    m, _ = move_to_submodule(m, g, ("conv", "mul", "relu", "add_1"), name)
    sub = m.get_submodule(name)
    sub_g = Tracer().trace(sub)
    sub_gm = GraphModule(sub, sub_g)
    expected = """


def forward(self, x):
    conv = self.conv(x);  x = None
    mul = conv * 2.0;  conv = None
    relu = self.relu(mul);  mul = None
    add = relu + 1;  relu = None
    return add
    """
    assert expected == str(sub_gm.code)


def test_replace_nodes_in_parent_module():
    m = Model()
    g = Tracer().trace(m)
    name = "new_sub"
    m, _ = move_to_submodule(m, g, ("conv", "mul", "relu", "add_1"), name)

    expected = """


def forward(self, x):
    add = x + 1;  x = None
    new_sub = self.new_sub(add);  add = None
    conv2 = self.conv2(new_sub);  new_sub = None
    return conv2
    """
    assert expected == str(m.code)


class TestFirstNodeUsesMoreThanOneInput:
    def test_create_correct_submodule_(self):
        m = Model()
        g = Tracer().trace(m)
        name = "new_sub"
        m, _ = move_to_submodule(m, g, ("mul", "relu", "add_1"), name)
        sub = m.get_submodule(name)
        sub_g = Tracer().trace(sub)
        sub_gm = GraphModule(sub, sub_g)
        expected = """


def forward(self, x):
    mul = x * 2.0;  x = None
    relu = self.relu(mul);  mul = None
    add = relu + 1;  relu = None
    return add
    """
        assert expected == str(sub_gm.code)

    def test_replace_nodes_in_parent_module(self):
        m = Model()
        g = Tracer().trace(m)
        name = "new_sub"
        m, _ = move_to_submodule(m, g, ("mul", "relu", "add_1"), name)

        expected = """


def forward(self, x):
    add = x + 1;  x = None
    conv = self.conv(add);  add = None
    new_sub = self.new_sub(conv);  conv = None
    conv2 = self.conv2(new_sub);  new_sub = None
    return conv2
    """
        assert expected == str(m.code)
