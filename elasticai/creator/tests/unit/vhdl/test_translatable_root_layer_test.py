import unittest

from elasticai.creator.vhdl.hw_equivalent_layers.layers import AbstractTranslatableLayer


class MyRootLayer(AbstractTranslatableLayer):
    pass


class TranslatableRootLayerTest(unittest.TestCase):
    """
    Use AbstractTranslatableLayer to inject a different tracer and return a simplified graph representation.
    I don't think we need to inherit from torch.nn.Module in AbstractTranslatableLayer then.
    """

    ...
