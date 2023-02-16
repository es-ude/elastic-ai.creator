from ._signal import Signal
from ._signal_impl import _SignalConfiguration, _SignalImpl
from ._vector_signal_impl import _VectorSignalConfiguration, _VectorSignalImpl


class SignalBuilder:
    def __init__(self):
        self.args = {
            "width": 0,
            "id": "x",
            "accepted_names": list(),
            "default": None,
        }

    def width(self, width: int) -> "SignalBuilder":
        self.args.update({"width": width})
        return self

    def id(self, name: str) -> "SignalBuilder":
        self.args.update({"id": name})
        return self

    def accepted_names(self, names: list[str]) -> "SignalBuilder":
        self.args.update({"accepted_names": names})
        return self

    def default(self, value: str) -> "SignalBuilder":
        self.args.update({"default": value})
        return self

    def build(self) -> Signal:
        if self.args["width"] > 0:
            return _VectorSignalImpl(_VectorSignalConfiguration(**self.args))
        else:
            args_without_width = filter(
                lambda entry: entry[0] != "width", self.args.items()
            )
            return _SignalImpl(_SignalConfiguration(**dict(args_without_width)))
