import unittest

import torch
from torch.nn import Conv2d

from elasticai.creator.qat.QRigL import QRigLScheduler


class test_QRigl(unittest.TestCase):
    def test_QrigL_initialization_check_mask(self):
        layer = Conv2d(in_channels=2, out_channels=2, kernel_size=3)
        scheduler = QRigLScheduler(
            weighted_layers=(layer,),
            optimizer=torch.optim.Adam([layer.weight]),
            num_connections=2,
            T_end=0,
        )
        self.assertSequenceEqual(layer.weight.shape, scheduler.backward_masks[0].shape)
        self.assertEqual(torch.count_nonzero(scheduler.backward_masks[0][0]), 2)

    def test_QrigL_weight_is_initially_multiplied_by_mask(self):
        layer = Conv2d(in_channels=4, out_channels=2, kernel_size=3)
        scheduler = QRigLScheduler(
            weighted_layers=(layer,),
            optimizer=torch.optim.Adam([layer.weight]),
            num_connections=2,
            T_end=0,
        )
        self.assertTrue(
            torch.all(layer.weight == layer.weight * scheduler.backward_masks[0])
        )

    def test_QrigL_weight_is_after_step_multiplied_by_mask(self):
        layer = Conv2d(in_channels=2, out_channels=2, kernel_size=3)
        optimizer = torch.optim.Adam([layer.weight])
        scheduler = QRigLScheduler(
            weighted_layers=(layer,), optimizer=optimizer, num_connections=2, T_end=0
        )
        new_mask = torch.zeros_like(layer.weight)
        scheduler.backward_masks[0] = new_mask
        optimizer.step()
        self.assertTrue(torch.all(layer.weight == new_mask))

    def test_QrigL_check_grad_accumulation(self):
        layer = Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        optimizer = torch.optim.Adam([layer.weight])
        scheduler = QRigLScheduler(
            weighted_layers=(layer,),
            optimizer=optimizer,
            num_connections=4,
            delta=4,
            grad_accumulation_n=2,
            T_end=100,
        )
        for i in range(3):
            optimizer.zero_grad()
            test_input = torch.ones((1, 2, 1, 1))
            out = layer(test_input)
            out.mean().backward()
            scheduler()
        n = scheduler.backward_hook_objects[0].dense_grad
        m = layer.weight.grad
        self.assertTrue(
            torch.all(
                layer.weight.grad / 2 == scheduler.backward_hook_objects[0].dense_grad
            )
        )

    def Qrigl_step_helper(self):
        layer = Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        scheduler = QRigLScheduler(
            weighted_layers=(layer,),
            optimizer=torch.optim.Adam([layer.weight]),
            num_connections=1,
            T_end=100,
            alpha=1,
        )
        layer.weight.grad = torch.ones_like(layer.weight)
        scheduler.backward_hook_objects[0].dense_grad = torch.zeros_like(layer.weight)
        scheduler.backward_hook_objects[0].dense_grad[0][0] = 1
        scheduler.backward_hook_objects[0].dense_grad[0][1] = 0
        scheduler.backward_masks[0][0][0] = 0
        scheduler.backward_masks[0][0][1] = 1
        return layer, scheduler

    def test_QrigL_rigl_after_step_mask(self):
        layer, scheduler = self.Qrigl_step_helper()
        scheduler._rigl_step()
        self.assertEqual(scheduler.backward_masks[0][0][0][0][0].item(), 1.0)
        self.assertEqual(scheduler.backward_masks[0][0][1][0][0].item(), 0.0)

    def test_QrigL_grad_is_after_step_multiplied_by_mask(self):
        layer, scheduler = self.Qrigl_step_helper()
        scheduler._rigl_step()
        expected_grad = torch.ones_like(layer.weight)
        expected_grad[0][1] = 0
        self.assertTrue(torch.all(layer.weight.grad == expected_grad))

    def test_QrigL_call(self):
        layer = Conv2d(in_channels=2, out_channels=2, kernel_size=3)
        scheduler = QRigLScheduler(
            weighted_layers=(layer,),
            optimizer=torch.optim.Adam([layer.weight]),
            num_connections=2,
            T_end=101,
            delta=10,
        )
        for call in range(200):
            layer.weight.grad = torch.zeros_like(layer.weight)
            scheduler()
        self.assertEqual(scheduler.step, 200)
        self.assertEqual(scheduler.rigl_steps, 10)

    def test_QrigL_momentum(self):
        layer = Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        optimizer = torch.optim.SGD([layer.weight], lr=0.01, momentum=0.1)
        scheduler = QRigLScheduler(
            weighted_layers=(layer,),
            optimizer=optimizer,
            num_connections=2,
            T_end=101,
            delta=10,
        )
        scheduler.backward_masks[0] = torch.zeros_like(layer.weight)
        scheduler.backward_masks[0][0][1] = 1
        for i in range(3):
            optimizer.zero_grad()
            test_input = torch.ones((1, 2, 1, 1))
            out = layer(test_input)
            out.mean().backward()
            scheduler()
            optimizer.step()
        buffer = optimizer.state[layer.weight]["momentum_buffer"]
        self.assertTrue(torch.all(buffer == buffer * scheduler.backward_masks[0]))

    def test_QrigL_state_dict(self):
        layer = Conv2d(in_channels=2, out_channels=2, kernel_size=3)
        scheduler = QRigLScheduler(
            weighted_layers=(layer,),
            optimizer=torch.optim.Adam([layer.weight]),
            num_connections=2,
            T_end=101,
            delta=10,
        )
        layer2 = Conv2d(in_channels=2, out_channels=2, kernel_size=3)
        scheduler2 = QRigLScheduler(
            weighted_layers=(layer2,),
            optimizer=torch.optim.Adam([layer.weight]),
            num_connections=0,
            T_end=0,
            delta=0,
            state_dict=scheduler.state_dict(),
        )
        a = scheduler == scheduler2
        self.assertTrue(scheduler == scheduler2)


if __name__ == "__main__":
    unittest.main()
