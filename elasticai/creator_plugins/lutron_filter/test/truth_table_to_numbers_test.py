from unittest import TestCase

import torch

from elasticai.creator_plugins.lutron_filter.nn.lutron.truth_table_generation import (
    decode_from_number_per_channel_to_bit_vector_per_input,
    decode_from_number_per_channel_to_bitstrings,
    encode_as_number_per_channel,
    encode_as_number_per_row,
)


class NumberEncodingTest(TestCase):
    def test_single_in_channel_single_out_channel(self):
        outs = torch.tensor([[0.0], [0.0]], dtype=torch.int64)
        expected = (0,)
        actual = tuple(encode_as_number_per_channel(outs))
        self.assertEqual(expected, actual)

    def test_double_in_channel_single_out_channel(self):
        outs = torch.tensor([[0], [1], [1], [0]])
        expected = [6]
        actual = list(encode_as_number_per_channel(outs))
        self.assertEqual(expected, actual)

    def test_single_in_channel_double_out_channel(self):
        outs = torch.tensor([[0, 1], [1, 0]])
        expected = [1, 2]
        actual = list(encode_as_number_per_channel(outs))
        self.assertEqual(expected, actual)

    def test_double_in_channel_double_out_channel(self):
        outs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        expected = [5, 3]
        actual = list(encode_as_number_per_channel(outs))
        self.assertEqual(expected, actual)

    def test_decode_to_bit_tuple(self):
        expected = ((0, 0), (1, 0), (0, 1), (1, 1))
        outs = [5, 3]
        actual = decode_from_number_per_channel_to_bit_vector_per_input(outs)
        self.assertEqual(expected, tuple(actual))

    def test_decode_to_bit_strings(self):
        expected = ("00", "10", "01", "11")
        outs = [5, 3]
        actual = decode_from_number_per_channel_to_bitstrings(outs, in_channels=2)
        self.assertEqual(expected, tuple(actual))

    def test_triple_in_single_out(self):
        expected = (2**7,)
        outs = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
        outs.resize_(8, 1)
        actual = encode_as_number_per_channel(outs)
        self.assertEqual(expected, tuple(actual))

        actual = tuple(encode_as_number_per_channel(outs))
        self.assertEqual(expected, actual)

    def test_raise_error_when_encoding_more_than_six_in_channels(self):
        def call():
            tuple(encode_as_number_per_channel(torch.zeros((65, 1))))

        self.assertRaises(
            ValueError,
            call,
        )

    def test_encoding_per_row(self):
        expected = (0, 3, 1, 2)
        outs = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
        actual = encode_as_number_per_row(outs)
        self.assertEqual(expected, tuple(actual))
