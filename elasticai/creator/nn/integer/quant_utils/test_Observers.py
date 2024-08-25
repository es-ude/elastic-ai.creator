import torch

from elasticai.creator.nn.integer.quant_utils.Observers import (
    GlobalMinMaxObserver,
    LocalMinMaxObserver,
    MovingAverageMinMaxObserver,
)


def setUp1_for_GlobalMinMaxObserver():
    # range gets larger
    tensor1 = torch.tensor([-1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    tensor2 = torch.tensor([-5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
    tensor3 = torch.tensor([-9.0, 10.0, 11.0, 12.0], dtype=torch.float32)
    return tensor1, tensor2, tensor3


def _setUp2_for_GlobalMinMaxObserver():
    # range gets smaller
    tensor1 = torch.tensor([-9.0, 10.0, 11.0, 12.0], dtype=torch.float32)
    tensor2 = torch.tensor([-5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
    tensor3 = torch.tensor([-1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    return tensor1, tensor2, tensor3


def test_GlobalMinMaxObserver_setUp1():
    tensor1, tensor2, tensor3 = setUp1_for_GlobalMinMaxObserver()
    observer = GlobalMinMaxObserver()

    observer(tensor1)
    assert torch.equal(observer.min_float, torch.tensor([-1.0], dtype=torch.float32))
    assert torch.equal(observer.max_float, torch.tensor([4.0], dtype=torch.float32))
    observer(tensor2)
    assert torch.equal(observer.min_float, torch.tensor([-5.0], dtype=torch.float32))
    assert torch.equal(observer.max_float, torch.tensor([8.0], dtype=torch.float32))
    observer(tensor3)
    assert torch.equal(observer.min_float, torch.tensor([-9.0], dtype=torch.float32))
    assert torch.equal(observer.max_float, torch.tensor([12.0], dtype=torch.float32))


def test_GlobalMinMaxObserver_setUp2():
    tensor1, tensor2, tensor3 = _setUp2_for_GlobalMinMaxObserver()
    observer = GlobalMinMaxObserver()
    observer(tensor1)
    assert torch.equal(observer.min_float, torch.tensor([-9.0], dtype=torch.float32))
    assert torch.equal(observer.max_float, torch.tensor([12.0], dtype=torch.float32))
    observer(tensor2)
    assert torch.equal(observer.min_float, torch.tensor([-9.0], dtype=torch.float32))
    assert torch.equal(observer.max_float, torch.tensor([12.0], dtype=torch.float32))
    observer(tensor3)
    assert torch.equal(observer.min_float, torch.tensor([-9.0], dtype=torch.float32))
    assert torch.equal(observer.max_float, torch.tensor([12.0], dtype=torch.float32))
