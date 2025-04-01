from abc import abstractmethod
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Generic, ParamSpec, Protocol, TypeVar

from elasticai.creator.function_utils import KeyedFunctionDispatcher as _Registry
from elasticai.creator.function_utils import RegisterDescriptor


class Lowerable(Protocol):
    @property
    @abstractmethod
    def type(self) -> str: ...


Tin = TypeVar("Tin", bound="Lowerable")
Tout = TypeVar("Tout")


class LoweringPass(Generic[Tin, Tout]):
    register: RegisterDescriptor[Tin, Tout] = RegisterDescriptor()
    register_iterable: RegisterDescriptor[Tin, Iterable[Tout]] = RegisterDescriptor()

    def __init__(self) -> None:
        def key_lookup_fn(x: Tin) -> str:
            return x.type

        self._fns: _Registry[Tin, Iterable[Tout]] = _Registry(key_lookup_fn)

    def _register_callback(self, name: str, fn: Callable[[Tin], Tout]):
        self._check_for_redefinition(name)
        wrapper = return_as_iterable(fn)
        self._fns.register(name)(wrapper)

    def _register_iterable_callback(
        self, name: str, fn: Callable[[Tin], Iterable[Tout]]
    ):
        self._check_for_redefinition(name)
        self._fns.register(name)(fn)

    def __call__(self, args: Iterable[Tin]) -> Iterable[Tout]:
        for arg in args:
            yield from self._fns(arg)

    def _check_for_redefinition(self, arg):
        if arg in self._fns:
            raise ValueError(f"function for {arg} already defined in lowering pass")


P = ParamSpec("P")


def return_as_iterable(fn: Callable[P, Tout]) -> Callable[P, Iterable[Tout]]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterable[Tout]:
        yield fn(*args, **kwargs)

    return wrapper
