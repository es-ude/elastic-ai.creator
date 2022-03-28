import unittest

from elasticai.creator.vhdl.generator.generator_functions import (
    precomputed_logic_function_process,
)
from elasticai.creator.vhdl.number_representations import BitVector


class test_generatorfunctions(unittest.TestCase):
    def test_precomputed_conv_function_process(self):
        bit_vector_x = [
            [
                BitVector(bit_width=1, number=1, repr=1),
                BitVector(bit_width=1, number=1, repr=1),
            ],
            [
                BitVector(bit_width=1, number=-1, repr=0),
                BitVector(bit_width=1, number=1, repr=1),
            ],
        ]
        bit_vector_y = [
            [BitVector(bit_width=1, number=-1, repr=0)],
            [BitVector(bit_width=1, number=1, repr=1)],
        ]
        expected = ['y <="0" when x="11" else', '"1" when x="01" ;']
        process = list(precomputed_logic_function_process(bit_vector_x, bit_vector_y))
        self.assertSequenceEqual(expected, process)


if __name__ == "__main__":
    unittest.main()
