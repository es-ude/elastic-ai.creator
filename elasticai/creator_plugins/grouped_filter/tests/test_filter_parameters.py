from unittest import TestCase

from elasticai.creator_plugins.grouped_filter import FilterParameters


class GetInChannelsByGroup(TestCase):
    def test_two_in_channels_in_two_groups_gets_0_1(self):
        p = FilterParameters(kernel_size=1, in_channels=2, out_channels=2, groups=2)
        self.assertEqual(((0,), (1,)), p.get_in_channels_by_group())

    def test_4_in_channels_2_groups_gets_01_23(self):
        p = FilterParameters(kernel_size=1, in_channels=4, out_channels=4, groups=2)
        self.assertEqual(((0, 1), (2, 3)), p.get_in_channels_by_group())
