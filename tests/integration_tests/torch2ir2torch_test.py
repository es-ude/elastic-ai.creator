import hypothesis.strategies as st
import pytest
from hypothesis import settings as hypothesis_settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)
from torch.nn import Conv1d, Flatten, Linear, ReLU, Sequential, Sigmoid

from elasticai.creator.ir2torch import get_default_converter as ir2torch
from elasticai.creator.torch2ir import get_default_converter as torch2ir


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def convert(model):
    return torch2ir().convert(model)


@st.composite
def conv_channel_group_parameters(draw: st.DrawFn) -> tuple[int, int, int]:
    test_class_instance = draw(st.runner())
    out_channels = draw(st.integers(min_value=1, max_value=64))
    in_channels = test_class_instance.last_out_channels
    if in_channels == -1:
        in_channels = draw(st.integers(max_value=64))
    groups = draw(
        st.integers(min_value=1, max_value=out_channels).filter(
            lambda n: out_channels % n == 0 and in_channels % n == 0
        )
    )
    return in_channels, out_channels, groups


@hypothesis_settings(max_examples=200, stateful_step_count=20)
class BuildModelFromIr(RuleBasedStateMachine):
    @initialize(start_channels=st.integers(min_value=1, max_value=64))
    def setup(self, start_channels):
        self.model = Sequential()
        self.last_out_channels = start_channels
        self.is_flattened = False

    @rule(
        channels_and_groups=conv_channel_group_parameters(),
        bias=st.booleans(),
        stride=st.integers(min_value=1, max_value=4),
        kernel_size=st.integers(min_value=1, max_value=12),
        padding=st.integers(min_value=0, max_value=3),
    )
    @precondition(lambda self: not self.is_flattened)
    def append_conv(self, channels_and_groups, bias, stride, kernel_size, padding):
        in_channels, out_channels, groups = channels_and_groups
        self.model.append(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                groups=groups,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        self.last_out_channels = out_channels

    @rule()
    def append_relu(self):
        self.model.append(ReLU())

    @rule()
    def append_sigmoid(self):
        self.model.append(Sigmoid())

    @rule()
    @precondition(lambda self: not self.is_flattened)
    def flatten(self):
        self.model.append(Flatten())
        self.is_flattened = True

    @rule(out_features=st.integers(min_value=1, max_value=32), bias=st.booleans())
    @precondition(lambda self: self.is_flattened)
    def append_linear(self, out_features, bias):
        self.model.append(
            Linear(
                in_features=self.last_out_channels, out_features=out_features, bias=bias
            )
        )

    @invariant()
    def model_from_ir_and_state_dict_is_the_same(self):
        original = self.model
        ir = convert(original)
        state = original.state_dict()
        rebuilt = ir2torch().convert(ir, state)

        def to_native_python(data) -> dict:
            result = {}
            for name, p in data:
                result[name] = p.tolist() if hasattr(p, "tolist") else p
            return result

        original_params = to_native_python(original.named_parameters())
        rebuilt_params = to_native_python(rebuilt.named_parameters())
        assert original_params == rebuilt_params
        del rebuilt


BuildModelFromIrTest = pytest.mark.slow()(BuildModelFromIr.TestCase)
