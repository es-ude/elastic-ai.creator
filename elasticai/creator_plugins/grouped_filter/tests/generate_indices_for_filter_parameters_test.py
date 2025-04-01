import unittest

from elasticai.creator_plugins.grouped_filter import FilterParameters
from elasticai.creator_plugins.grouped_filter.src.index_generators import (
    GroupedFilterIndexGenerator,
)


def to_tuple(xs):
    if hasattr(xs, "__iter__"):
        return tuple(to_tuple(x) for x in xs)
    return xs


class UnrolledFilterParameterIndicesTest(unittest.TestCase):
    def test_respects_channelwise_interleaving_for_steps(self):
        expected = (((0, 1),), ((2, 3),))
        actual = GroupedFilterIndexGenerator(
            params=FilterParameters(
                input_size=2,
                in_channels=2,
                kernel_size=1,
                groups=1,
                stride=1,
                out_channels=1,
            ),
        ).as_tuple_by_steps()
        self.assertEqual(expected, actual)

    def test_using_two_groups_for_two_channels_separates_channels(self):
        expected = (((0,), (1,)), ((2,), (3,)))
        actual = GroupedFilterIndexGenerator(
            params=FilterParameters(
                input_size=2,
                in_channels=2,
                kernel_size=1,
                groups=2,
                stride=1,
                out_channels=2,
            ),
        ).as_tuple_by_steps()
        self.assertEqual(expected, actual)

    def test_generate_3_steps_for_spatial_size_5_and_window_size_3(self):
        indices = GroupedFilterIndexGenerator(
            params=FilterParameters(
                input_size=5,
                kernel_size=3,
                in_channels=1,
                groups=1,
                stride=1,
                out_channels=1,
            ),
        ).as_tuple_by_steps()
        self.assertEqual(3, len(indices))

    def test_time_steps_from_same_channel_are_concatenated_across_window(self):
        expected = (
            (
                (
                    0,
                    2,
                ),
                (
                    1,
                    3,
                ),
            ),
        )
        indices = GroupedFilterIndexGenerator(
            params=FilterParameters(
                input_size=2,
                kernel_size=2,
                in_channels=2,
                groups=2,
                stride=1,
                out_channels=2,
            ),
        ).as_tuple_by_steps()
        self.assertEqual(expected, indices)

    def test_strides_are_correctly_concatenated(self):
        expected = (((0, 1, 4, 5), (2, 3, 6, 7)), ((4, 5, 8, 9), (6, 7, 10, 11)))
        generator = GroupedFilterIndexGenerator(
            params=FilterParameters(
                input_size=3,
                kernel_size=2,
                in_channels=4,
                groups=2,
                stride=1,
                out_channels=2,
            ),
        )
        indices = generator.as_tuple_by_steps()

        self.assertEqual(expected, indices)

    def test_index_generator_interface(self):
        expected = [[[0, 1, 4, 5], [2, 3, 6, 7]], [[4, 5, 8, 9], [6, 7, 10, 11]]]
        generator = GroupedFilterIndexGenerator(
            params=FilterParameters(
                input_size=3,
                kernel_size=2,
                in_channels=4,
                groups=2,
                stride=1,
                out_channels=2,
            ),
        )
        actual = []
        for step in generator.steps():
            groups = []
            for group in step.groups():
                indices = list(group)
                groups.append(indices)
            actual.append(groups)

        self.assertEqual(expected, actual)

    def test_index_generator_interface_by_groups(self):
        expected = [
            [[0, 1, 4, 5], [4, 5, 8, 9]],
            [[2, 3, 6, 7], [6, 7, 10, 11]],
        ]
        generator = GroupedFilterIndexGenerator(
            params=FilterParameters(
                input_size=3,
                kernel_size=2,
                in_channels=4,
                groups=2,
                stride=1,
                out_channels=2,
            ),
        )
        actual = []
        for group in generator.groups():
            steps = []
            for step in group.steps():
                indices = list(step)
                steps.append(indices)
            actual.append(steps)

        self.assertEqual(expected, actual)
