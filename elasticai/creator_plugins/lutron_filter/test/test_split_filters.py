from functools import partial
from unittest import TestCase

from elasticai.creator_plugins.lutron_filter.splitting_filters.split_filter import (
    find_channel_sizes_from_g1_to_g2,
    split_by_group_end_with_pointwise,
)

from ._imports import FilterParameters


class ParamListBaseTest(TestCase):
    def check_param_lists(self, expected, actual):
        for idx in range(len(expected)):
            with self.subTest(f"filter number: {idx}"):
                self.assertEquals(expected[idx], actual[idx])


class TestSplitConfigs(ParamListBaseTest):
    def check_splits(self, expected, actual):
        for idx in range(len(expected)):
            with self.subTest(f"split config number: {idx}"):
                self.assertEquals(expected[idx], actual[idx])

    def test_counting_split_channels_up_to_max_fan_in_for_single_in_out_channels(self):
        fp = partial(FilterParameters, kernel_size=1)
        split = split_by_group_end_with_pointwise(
            fp(in_channels=1, out_channels=1), max_fan_in=6
        )
        expected = [
            (fp(in_channels=1, out_channels=n), fp(in_channels=n, out_channels=1))
            for n in range(1, 7)
        ]
        self.assertEqual(set(expected), split)

    def test_input_channel_2(self):
        def fp(in_channels, out_channels, groups=1):
            return FilterParameters(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
            )

        split = split_by_group_end_with_pointwise(
            fp(in_channels=2, out_channels=1), max_fan_in=6
        )
        expected = [
            (
                fp(in_channels=2, out_channels=n, groups=1),
                fp(in_channels=n, out_channels=1, groups=1),
            )
            for n in range(1, 7)
        ] + [
            (
                fp(in_channels=2, out_channels=2, groups=2),
                fp(in_channels=2, out_channels=1, groups=1),
            ),
            (
                fp(in_channels=2, out_channels=4, groups=2),
                fp(in_channels=4, out_channels=1, groups=1),
            ),
            (
                fp(in_channels=2, out_channels=6, groups=2),
                fp(in_channels=6, out_channels=1, groups=1),
            ),
        ]
        self.assertEqual(set(expected), split)

    def test_in_1_out_2(self):
        def fp(in_channels, out_channels, groups=1):
            return FilterParameters(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
            )

        split = split_by_group_end_with_pointwise(
            fp(in_channels=1, out_channels=2), max_fan_in=6
        )
        expected = {
            (
                fp(in_channels=1, out_channels=n, groups=1),
                fp(in_channels=n, out_channels=2, groups=1),
            )
            for n in range(1, 7)
        } | {
            (
                fp(in_channels=1, out_channels=2, groups=1),
                fp(in_channels=2, out_channels=2, groups=2),
            ),
            (
                fp(in_channels=1, out_channels=4, groups=1),
                fp(in_channels=4, out_channels=2, groups=2),
            ),
            (
                fp(in_channels=1, out_channels=6, groups=1),
                fp(in_channels=6, out_channels=2, groups=2),
            ),
            (
                fp(in_channels=1, out_channels=8, groups=1),
                fp(in_channels=8, out_channels=2, groups=2),
            ),
            (
                fp(in_channels=1, out_channels=10, groups=1),
                fp(in_channels=10, out_channels=2, groups=2),
            ),
            (
                fp(in_channels=1, out_channels=12, groups=1),
                fp(in_channels=12, out_channels=2, groups=2),
            ),
        }
        self.assertEqual(expected, split)

    def test_6_in_6_out_max_fan_in_3(self):
        def fp(in_channels, out_channels, groups=1):
            return FilterParameters(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                groups=groups,
            )

        ic = 6
        oc = 6
        split = split_by_group_end_with_pointwise(
            fp(in_channels=ic, out_channels=oc), max_fan_in=3
        )

        def pair(g1, sc, g2):
            return (
                fp(in_channels=ic, out_channels=sc, groups=g1),
                fp(in_channels=sc, out_channels=oc, groups=g2),
            )

        expected = {
            pair(g1=2, sc=2, g2=1),
            pair(g1=2, sc=2, g2=2),
            pair(g1=2, sc=4, g2=2),
            pair(g1=2, sc=6, g2=2),
            pair(g1=2, sc=6, g2=3),
            pair(g1=2, sc=6, g2=6),
            pair(g1=2, sc=12, g2=6),
            pair(g1=2, sc=18, g2=6),
            pair(g1=3, sc=3, g2=1),
            pair(g1=3, sc=6, g2=2),
            pair(g1=3, sc=3, g2=3),
            pair(g1=3, sc=6, g2=3),
            pair(g1=3, sc=9, g2=3),
            pair(g1=3, sc=6, g2=6),
            pair(g1=3, sc=12, g2=6),
            pair(g1=3, sc=18, g2=6),
            pair(g1=6, sc=6, g2=2),
            pair(g1=6, sc=6, g2=3),
            pair(g1=6, sc=6, g2=6),
            pair(g1=6, sc=12, g2=6),
            pair(g1=6, sc=18, g2=6),
        }
        self.assertEqual(expected, split)

    def test_build_split_channel_set(self):
        with self.subTest(g1=6, g2=3, max_fanin=3):
            self.assertEqual({6}, find_channel_sizes_from_g1_to_g2(6, 3, 3, 1))
        with self.subTest(g1=3, g2=6, max_fanin=3):
            self.assertEqual({6, 12, 18}, find_channel_sizes_from_g1_to_g2(3, 6, 3, 1))
