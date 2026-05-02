import elasticai.creator_plugins.lutron_filter as lf
from elasticai.creator_plugins.lutron_filter.utils import (
    NetworkBuilder,
    batchnorm1d,
    binarize,
    conv1d,
    linear,
    maxpool,
)

from elasticai.creator.ir2vhdl import Shape


def test_precomp_lin():
    builder = NetworkBuilder()
    lin = builder.add(
        linear(in_features=1, out_features=1, weight=[[-0.5]], use_bias=False)
    )
    bin = builder.add(binarize())
    x = builder.open(Shape(1, 1))
    x = bin(x)
    x = lin(x)
    x = bin(x)
    reg = builder.close(x)
    g, reg = lf.precompute(reg["root"], reg)
    assert set((n.type for n in g.nodes.values())) == {
        "filter",
        "input",
        "output",
    }
    assert reg["lutron"].attributes["truth_table"] == (
        ("0", "1"),
        ("1", "0"),
    )


def test_precomp_maxpool():
    builder = NetworkBuilder()
    maxp = builder.add(maxpool(2, 2))
    bin = builder.add(binarize())
    x = builder.open(Shape(1, 4))
    x = bin(x)
    x = maxp(x)
    x = bin(x)
    reg = builder.close(x)
    g, reg = lf.precompute(reg["root"], reg)
    assert "lutron_filter" in set(k for k in g.nodes)
    assert (
        g.nodes["lutron_filter"].attributes["filter_parameters"].get_int("kernel_size")
        == 2
    )
    assert reg["lutron"].attributes["truth_table"] == (
        ("00", "0"),
        ("01", "1"),
        ("10", "1"),
        ("11", "1"),
    )


def test_precomp_conv1d():
    builder = NetworkBuilder()
    conv = builder.add(
        conv1d(
            kernel_size=2,
            in_channels=1,
            out_channels=2,
            use_bias=False,
            weight=[
                [[1.0, 0.5]],
                [[-1.0, 0.5]],
            ],
        )
    )
    bin = builder.add(binarize())
    x = builder.open(Shape(2, 4))
    x = bin(conv(bin(x)))
    reg = builder.close(x)
    g, reg = lf.precompute(reg["root"], reg)
    assert "lutron_filter" in set(k for k in g.nodes)
    assert "lutron" in set(k for k in reg.keys())
    assert set(reg["lutron"].attributes["truth_table"]) == set(
        (
            ("00", "01"),
            ("01", "01"),
            ("10", "10"),
            ("11", "10"),
        )
    )


def test_precomp_conv1d_with_bnorm():
    builder = NetworkBuilder()
    conv = builder.add(
        conv1d(
            kernel_size=2,
            in_channels=1,
            out_channels=2,
            use_bias=False,
            weight=[
                [[1.0, 0.5]],
                [[-1.0, 0.5]],
            ],
        )
    )
    bnorm = builder.add(
        batchnorm1d(
            num_features=2,
            running_mean=[0.6, 0.6],
            running_var=[1.0, 1.0],
        )
    )
    bin = builder.add(binarize())
    x = builder.open(Shape(2, 1))
    x = bin(bnorm(conv(bin(x))))
    reg = builder.close(x)
    g, reg = lf.precompute(reg["root"], reg)
    assert "lutron_filter" in set(k for k in g.nodes)
    assert "lutron" in set(k for k in reg.keys())
    assert set(reg["lutron"].attributes["truth_table"]) == set(
        (
            ("00", "00"),
            ("01", "01"),
            ("10", "00"),
            ("11", "10"),
        )
    )
