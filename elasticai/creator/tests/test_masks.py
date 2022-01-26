import unittest

from elasticai.creator.masks import fixedOffsetMask4D, randomMask4D
import numpy as np
import torch
class test_masks(unittest.TestCase):
    def test_fixedOffsetMask4D_channel_offset(self):
        mask = fixedOffsetMask4D(out_channels=4, kernel_size=(1, 1),
                                 in_channels=2, groups=1, channel_width=1, offset_axis=1)
        expected = torch.zeros((4,2,1,1))
        expected[0,0,0,0] = 1
        expected[1, 1, 0, 0] = 1
        expected[2,0,0,0] = 1
        expected[3, 1, 0, 0] = 1
        self.assertTrue(torch.all(torch.eq(mask,expected)))
    
    def test_fixedOffsetMask4D_channel_offset_width2(self):
        mask = fixedOffsetMask4D(out_channels=2, kernel_size=(1, 1),
                                 in_channels=4, groups=1, channel_width=2, offset_axis=1)
        expected = torch.zeros((2,4,1,1))
        expected[0,0,0,0] = 1
        expected[0, 1, 0, 0] = 1
        expected[1, 2, 0, 0] = 1
        expected[1, 3, 0, 0] = 1
        self.assertTrue(torch.all(torch.eq(mask,expected)))
    
    def test_fixedOffsetMask4D_axis_offset(self):
        mask = fixedOffsetMask4D(out_channels=4, kernel_size=(2, 1),
                                 in_channels=2, groups=1, channel_width=1, offset_axis=2)
        expected = torch.zeros((4,2,2,1))
        expected[0,0,0,0] = 1
        expected[0, 1, 0, 0] = 1
        expected[1, 0, 1, 0] = 1
        expected[1, 1, 1, 0] = 1
        expected[2, 0, 0, 0] = 1
        expected[2, 1, 0, 0] = 1
        expected[3, 0, 1, 0] = 1
        expected[3, 1, 1, 0] = 1
        self.assertTrue(torch.all(torch.eq(mask,expected)))
    
    def test_randomMask4D(self):
        mask = randomMask4D(out_channels=4, kernel_size=(2, 2),
                                 in_channels=2, groups=1,params_per_channel=2 )
        for channel in range(4):
            self.assertEqual(torch.count_nonzero(mask[channel]),2)

if __name__ == '__main__':
    unittest.main()
