import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot3d_against_fan_ins(filters, fn, name):
    np.linspace(0, 1, 256)
    np.linspace(1, 0, 256)
    np.linspace(0, 0, 256)
    LinearSegmentedColormap(
        "fan_in",
        {
            "red": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            "green": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
            "blue": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        },
        N=256,
    )
    in_groups = []
    out_groups = []
    metric = []
    for f in filters:
        in_groups.append(f[0].fan_in)
        out_groups.append(f[1].fan_in)
        metric.append(fn(f))

    plt.figure()
    ax = plt.subplot(projection="3d")
    ax.scatter(in_groups, out_groups, metric)
    ax.set_xlabel("fan in front")
    ax.set_ylabel("fan in back")
    ax.set_zlabel(name)
