from elasticai.creator.hdl.design_base.signal import Signal


def x(width: int) -> Signal:
    return Signal(name="x", width=width, accepted_names=["x", "y"])


def done() -> Signal:
    return Signal(name="done", width=0, accepted_names=["done", "enable"])


def enable() -> Signal:
    return Signal(name="enable", width=0, accepted_names=["done", "enable"])


def clock() -> Signal:
    return Signal(name="clock", width=0, accepted_names=["clock"])


def y(width: int) -> Signal:
    return Signal(name="y", width=width, accepted_names=["x", "y"])


def x_address(width: int) -> Signal:
    return Signal(
        name="x_address", width=width, accepted_names=["y_address", "x_address"]
    )


def y_address(width: int) -> Signal:
    return Signal(
        name="y_address", width=width, accepted_names=["x_address", "y_address"]
    )
