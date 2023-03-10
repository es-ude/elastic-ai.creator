from elasticai.creator.hdl.design_base.signal import Signal


def x(width: int) -> Signal:
    return Signal(name="x", width=width)


def done() -> Signal:
    return Signal(name="done", width=0)


def enable() -> Signal:
    return Signal(name="enable", width=0)


def clock() -> Signal:
    return Signal(name="clock", width=0)


def y(width: int) -> Signal:
    return Signal(name="y", width=width)


def x_address(width: int) -> Signal:
    return Signal(name="x_address", width=width)


def y_address(width: int) -> Signal:
    return Signal(name="y_address", width=width)
