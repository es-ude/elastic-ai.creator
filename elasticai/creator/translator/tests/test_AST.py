from unittest import TestCase

from torch.nn import Sequential

from elasticai.creator.layers import Binarize, QConv1d
from elasticai.creator.translator.parser import Parser


class CreatingASTs(TestCase):
    """
    Test List:
    - Bin-Conv-Bin yields precomputable block
    - input shape is inferrable from Bin-Conv-Bin block
      (This is probably more than one test)
    - Bin-Conv-Bin-Conv-Bin yields two precomputable blocks
    - Bin-Conv-BatchNorm-Bin yields one precomputable block
    - Bin-MaxPool-Conv-Bin yields one precomputable block
    - Bin-Linear-Bin yields no precomputable block
    - Bin-Conv-BatchNorm-Conv-Bin yields one precomputable block
    - We can access and execute the underlying pytorch layers
      corresponding to a block (here we would use our existing code again)
    """

    def test_empty_sequence_yields_empty_ast(self):
        model = Sequential()
        parser = Parser()
        ast = parser.parse(model)
        self.assertEqual(0, len(ast))

    @staticmethod
    def build_minimal_one_block_model():
        return Sequential(Binarize(),
                          QConv1d(in_channels=1,
                                  out_channels=1,
                                  kernel_size=(1,),
                                  stride=(1,),
                                  quantizer=Binarize()),
                          Binarize())

    def test_bin_conv_bin_yields_ast_of_length_one(self):
        model = self.build_minimal_one_block_model()
        parser = Parser()
        ast = parser.parse(model)
        self.assertEqual(1, len(ast))