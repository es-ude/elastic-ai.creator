import pytest
import torch


@pytest.fixture
def inputs():
    """Create sample input tensor for testing."""
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def relu_layer(tmp_path):
    """Create a ReLU layer for testing."""
    from elasticai.creator.nn.integer.MPQ.relu.relu import ReLU

    layer = ReLU(
        name="relu_0",
        quant_data_dir=str(tmp_path),
        device="cpu",
    )
    quant_configs = {
        "relu_0.inputs": 8,
        "relu_0.outputs": 8,
    }
    layer.set_quant_bits_from_config(quant_configs)
    return layer


@pytest.fixture
def linear_layer(tmp_path):
    """Create a Linear layer for testing."""
    from elasticai.creator.nn.integer.MPQ.linear.linear import Linear

    layer = Linear(
        name="linear_0",
        in_features=3,
        out_features=2,
        bias=True,
        quant_data_dir=str(tmp_path),
        device="cpu",
    )
    quant_configs = {
        "linear_0.inputs": 8,
        "linear_0.outputs": 8,
        "linear_0.weights": 8,
        "linear_0.bias": 8,
    }
    layer.set_quant_bits_from_config(quant_configs)
    layer.eval()
    return layer


@pytest.fixture
def sequential_layer(tmp_path):
    """Create a Sequential layer for testing."""
    from elasticai.creator.nn.integer.MPQ.linear.linear import Linear
    from elasticai.creator.nn.integer.MPQ.relu.relu import ReLU
    from elasticai.creator.nn.integer.MPQ.sequential.sequential import Sequential

    layer = Sequential(
        name="sequential_0",
        layers=[
            Linear(
                name="linear_0",
                in_features=3,
                out_features=4,
                quant_data_dir=str(tmp_path),
                device="cpu",
            ),
            ReLU(
                name="relu_0",
                quant_data_dir=str(tmp_path),
                device="cpu",
            ),
            Linear(
                name="linear_1",
                in_features=4,
                out_features=2,
                quant_data_dir=str(tmp_path),
                device="cpu",
            ),
        ],
    )
    quant_configs = {
        "linear_0.inputs": 8,
        "linear_0.outputs": 8,
        "linear_0.weights": 8,
        "linear_0.bias": 8,
        "linear_1.inputs": 8,
        "linear_1.outputs": 8,
        "linear_1.weights": 8,
        "linear_1.bias": 8,
        "relu_0.inputs": 8,
        "relu_0.outputs": 8,
    }
    layer.set_quant_bits_from_config(quant_configs)
    return layer


@pytest.fixture
def linear_layer_0(tmp_path):
    """Create the first Linear layer for sequential tests."""
    from elasticai.creator.nn.integer.MPQ.linear.linear import Linear

    layer = Linear(
        name="linear_0",
        in_features=3,
        out_features=4,
        bias=True,
        num_dimensions=1,
        quant_data_dir=str(tmp_path),
        device="cpu",
        MPQ_strategy="inheritance",
    )
    quant_configs = {
        "linear_0.inputs": 8,
        "linear_0.outputs": 8,
        "linear_0.weights": 8,
        "linear_0.bias": 8,
    }
    layer.set_quant_bits_from_config(quant_configs)
    layer.eval()
    return layer


@pytest.fixture
def relu_layer_0(tmp_path):
    """Create a ReLU layer for sequential tests."""
    from elasticai.creator.nn.integer.MPQ.relu.relu import ReLU

    layer = ReLU(
        name="relu_0",
        quant_data_dir=str(tmp_path),
        device="cpu",
        MPQ_strategy="inheritance",
    )
    quant_configs = {
        "relu_0.inputs": 8,
        "relu_0.outputs": 8,
    }
    layer.set_quant_bits_from_config(quant_configs)
    return layer


@pytest.fixture
def linear_layer_1(tmp_path):
    """Create the second Linear layer for sequential tests."""
    from elasticai.creator.nn.integer.MPQ.linear.linear import Linear

    layer = Linear(
        name="linear_1",
        in_features=4,
        out_features=2,
        bias=True,
        num_dimensions=1,
        quant_data_dir=str(tmp_path),
        device="cpu",
        MPQ_strategy="inheritance",
    )
    quant_configs = {
        "linear_1.inputs": 8,
        "linear_1.outputs": 8,
        "linear_1.weights": 8,
        "linear_1.bias": 8,
    }
    layer.set_quant_bits_from_config(quant_configs)
    layer.eval()
    return layer
