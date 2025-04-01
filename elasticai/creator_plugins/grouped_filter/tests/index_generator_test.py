from itertools import chain
from unittest import TestCase

from elasticai.creator_plugins.grouped_filter.src.index_generators import (
    generate_deinterleaved_indices,
    sliding_window,
)


class InterleaveChannelsAfterGroupwiseOrdering(TestCase):
    def select(self, filtered, indices):
        filtered = tuple(filtered)
        for i in indices:
            yield filtered[i]

    def test_interleave_3bit_output_operations(self):
        # running imaginary filters with kernel size 2 and 3 outputs, two filters per group produces
        expected_indices = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
        self.assertEqual(
            expected_indices, list(generate_deinterleaved_indices(12, 2, 3))
        )

    def test_interleave_2bit_output_operations(self):
        expected = tuple(range(8))
        groupwise_output = tuple(chain((0, 1, 4, 5), (2, 3, 6, 7)))
        interleaving_indices = generate_deinterleaved_indices(8, 2, 2)

        self.assertEqual(
            expected,
            tuple(self.select(filtered=groupwise_output, indices=interleaving_indices)),
        )

    def test_interleave_2bit_output_operation_with_length_16_and_8_channels(self):
        expected = tuple(range(16))
        groupwise_output = tuple(
            chain((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15))
        )
        interleaving_indices = tuple(generate_deinterleaved_indices(16, 2, 4))
        interleaved = tuple(self.select(groupwise_output, interleaving_indices))
        self.assertEqual(expected, tuple(interleaved))

    def prepare(self, filtered, channels, channel_size):
        filtered = tuple(chain(filtered))
        length = len(filtered)
        indices = tuple(generate_deinterleaved_indices(length, channels, channel_size))
        return tuple(range(length)), tuple(
            self.select(
                filtered,
                indices,
            )
        )

    def test_interleave_2bit_output_with_length_16_and_4_channels(self):
        tuple(chain((0, 1, 4, 5, 8, 9, 12, 13), (2, 3, 6, 7, 10, 11, 14, 15)))
        expected_indices = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)
        self.assertEqual(
            expected_indices, tuple(generate_deinterleaved_indices(16, 2, 2))
        )

    def test_interleave_2bit_output_with_length_12_and_3_channels(self):
        tuple(chain((0, 1, 6, 7), (2, 3, 8, 9), (4, 5, 10, 11)))
        expected_indices = (0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11)
        self.assertEqual(
            expected_indices, tuple(generate_deinterleaved_indices(12, 3, 2))
        )

    def test_interleave_3bit_output_with_length_18_and_3_channels(self):
        tuple(chain((0, 1, 2, 9, 10, 11), (3, 4, 5, 12, 13, 14), (6, 7, 8, 15, 16, 17)))
        expected_indices = (
            0,
            1,
            2,
            6,
            7,
            8,
            12,
            13,
            14,
            3,
            4,
            5,
            9,
            10,
            11,
            15,
            16,
            17,
        )
        self.assertEqual(
            expected_indices, tuple(generate_deinterleaved_indices(18, 3, 3))
        )

    def test_interleave_3bit_output_with_length_27_and_3_channels(self):
        groupwise_output = tuple(
            chain(
                (0, 1, 2, 9, 10, 11, 18, 19, 20),
                (3, 4, 5, 12, 13, 14, 21, 22, 23),
                (6, 7, 8, 15, 16, 17, 24, 25, 26),
            )
        )

        self.assertEqual(*self.prepare(groupwise_output, 3, 3))

    def test_interleave_3bit_output_with_length_24_and_2_channels(self):
        groupwise_output = tuple(
            chain(
                (0, 1, 2, 12, 13, 14),
                (3, 4, 5, 15, 16, 17),
                (6, 7, 8, 18, 19, 20),
                (9, 10, 11, 21, 22, 23),
            )
        )
        self.assertEqual(*self.prepare(groupwise_output, 4, 3))


class SlidingWindowIndicesTest(TestCase):
    def get_as_tuple(self, steps, size, stride):
        return tuple(tuple(x) for x in sliding_window(steps, size, stride))

    def test_two_steps(self):
        expected = ((0, 1, 2), (1, 2, 3))
        self.assertEqual(expected, self.get_as_tuple(2, 3, 1))

    def test_stride_2_skips_one_index(self):
        expected = ((0, 1, 2), (2, 3, 4))
        self.assertEqual(expected, self.get_as_tuple(2, 3, 2))

    def test_size_2_stride_3_misses_one_index(self):
        expected = ((0, 1), (3, 4))
        self.assertEqual(expected, self.get_as_tuple(2, 2, 3))
