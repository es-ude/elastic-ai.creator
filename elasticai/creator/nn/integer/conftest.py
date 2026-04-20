import pytest
import torch


@pytest.fixture
def inputs():
    """Create sample input tensor for testing."""
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def relu_layer(tmp_path):
    """Create a ReLU layer for testing."""
    from elasticai.creator.nn.integer.relu.relu import ReLU

    layer = ReLU(
        name="relu_0",
        quant_bits=8,
        quant_data_dir=str(tmp_path),
        device="cpu",
    )
    # Set to eval mode to prevent forward() from updating quant params
    layer.eval()
    # Set the zero_point to match expected THCRESHOLD value in test
    # THRESHOLD = int(zero_point) = -43
    layer.inputs_QParams.zero_point = torch.tensor([-43], dtype=torch.int32)
    return layer


@pytest.fixture
def linear_layer(tmp_path):
    """Create a Linear layer for testing."""
    from elasticai.creator.nn.integer.linear.linear import Linear

    layer = Linear(
        name="linear_0",
        quant_bits=8,
        in_features=3,
        out_features=2,
        device="cpu",
        bias=True,
        quant_data_dir=str(tmp_path),
    )
    # Set to eval mode to prevent forward() from updating quant params
    layer.eval()
    return layer


@pytest.fixture
def sequential_layer(tmp_path):
    """Create a Sequential layer for testing."""
    from elasticai.creator.nn.integer.linear.linear import Linear
    from elasticai.creator.nn.integer.relu.relu import ReLU
    from elasticai.creator.nn.integer.sequential.sequential import Sequential

    return Sequential(
        name="sequential_0",
        quant_bits=8,
        layers=[
            Linear(
                name="linear_0",
                quant_bits=8,
                in_features=3,
                out_features=4,
                device="cpu",
                quant_data_dir=str(tmp_path),
            ),
            ReLU(
                name="relu_0",
                quant_bits=8,
                quant_data_dir=str(tmp_path),
                device="cpu",
            ),
            Linear(
                name="linear_1",
                quant_bits=8,
                in_features=4,
                out_features=2,
                device="cpu",
                quant_data_dir=str(tmp_path),
            ),
        ],
    )


@pytest.fixture
def linear_layer_0(tmp_path):
    """Create the first Linear layer for sequential tests."""
    from elasticai.creator.nn.integer.linear.linear import Linear

    layer = Linear(
        name="linear_0",
        quant_bits=8,
        in_features=3,
        out_features=4,
        device="cpu",
        bias=True,
        quant_data_dir=str(tmp_path),
    )
    layer.eval()
    return layer


@pytest.fixture
def relu_layer_0(tmp_path):
    """Create a ReLU layer for sequential tests."""
    from elasticai.creator.nn.integer.relu.relu import ReLU

    return ReLU(
        name="relu_0",
        quant_bits=8,
        quant_data_dir=str(tmp_path),
        device="cpu",
    )


@pytest.fixture
def linear_layer_1(tmp_path):
    """Create the second Linear layer for sequential tests."""
    from elasticai.creator.nn.integer.linear.linear import Linear

    layer = Linear(
        name="linear_1",
        quant_bits=8,
        in_features=4,
        out_features=2,
        device="cpu",
        bias=True,
        quant_data_dir=str(tmp_path),
    )
    layer.eval()
    return layer
