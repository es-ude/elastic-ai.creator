from collections.abc import Iterable
from functools import wraps
from typing import Callable, Generic, ParamSpec, TypeVar, overload

Tin = TypeVar("Tin")
Tout = TypeVar("Tout")
P = ParamSpec("P")
FN = TypeVar("FN", bound=Callable)


def return_as_iterable(fn: Callable[P, Tout]) -> Callable[P, Iterable[Tout]]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterable[Tout]:
        yield fn(*args, **kwargs)

    return wrapper


class FnRegisterer(Generic[FN]):
    """Apply a callback to functions and either their own or custom names.


    Parameters
    ----------
    register_fn : will be called as `register_fn(name, fn)`


    Example
    -------
    >>> registry = dict()
    >>> def register_fn(name, fn):
    ...  registry[name] = fn
    ...
    >>> register = FnRegisterer(register_fn)
    >>> def my_fn(x):
    ...   print(x)
    ...
    >>> register(my_fn)
    >>> registry["my_fn"]("hello, world!")
    "hello, world!"


    another example could look like this

    ```python
    registry = dict()
    def register_fn(name, fn):
      registry[name] = fn

    register = FnRegisterer(register_fn)

    @register("other_name")
    @register
    def my_fn(x):
      print(x)
    ```

    This will add `my_fn` to the registry using the
    name `"my_fn"` and the name `"other_name"`.

    """

    def __init__(self, register_fn: Callable[[str, FN], None]):
        self._cb = register_fn

    @overload
    def __call__(self, name: str, /) -> Callable[[FN], FN]:
        """Return a function that can be used to register another function with `name`."""
        ...

    @overload
    def __call__(self, fn: FN, /) -> FN:
        """Register `fn` using its own name."""
        ...

    def __call__(self, arg: FN | str, /) -> FN | Callable[[FN], FN]:
        if isinstance(arg, str):
            return self.__reg_by_name(arg)
        return self.__reg(arg)

    def __reg_by_name(self, name: str) -> Callable[[FN], FN]:
        def reg(fn: FN) -> FN:
            self._cb(name, fn)
            return fn

        return reg

    def __reg(self, fn: FN) -> FN:
        self._cb(fn.__name__, fn)
        return fn


FNcon = TypeVar("FNcon", contravariant=True, bound=Callable)


class RegisterDescriptor(Generic[Tin, Tout]):
    """Automatically connect the `FnRegisterer` to a callback and make it look like a method.

    The owning instance needs to define a callback that has the name `f"_{name}_callback"`,
    where `name` is the name of the field assigned to this descriptor.

    For an example see the `MultiArgDispatcher` below.
    """

    def __set_name__(self, instance, name):
        self._cb_name = f"_{name}_callback"

    def __get__(self, instance, owner=None) -> FnRegisterer[Callable[[Tin], Tout]]:
        return FnRegisterer(getattr(instance, self._cb_name))


class MultiArgDispatcher(Generic[Tin, Tout]):
    """
    Functions can be registered with custom key or by
    using the functions name as key.
    Upon `call` the registry will use the function
    specified by the `dispatch_key_fn` and the argument
    to `call` to process the argument and return the result.
    """

    register: RegisterDescriptor[Tin, Tout] = RegisterDescriptor()

    def __init__(self, dispatch_key_fn: Callable[[Tin], str]) -> None:
        self._key_fn = dispatch_key_fn
        self._fns: dict[str, Callable[[Tin], Tout]] = dict()

    def _register_callback(self, name: str, fn: Callable[[Tin], Tout]) -> None:
        self._fns[name] = fn

    def __contains__(self, item: str | Callable[[Tin], Tout]) -> bool:
        if isinstance(item, str):
            return item in self._fns
        return item.__name__ in self._fns

    def can_dispatch(self, item: Tin) -> bool:
        return self._key_fn(item) in self

    def call(self, arg: Tin) -> Tout:
        return self._fns[self._key_fn(arg)](arg)

    def __call__(self, arg: Tin) -> Tout:
        return self.call(arg)


class FunctionRegistry(MultiArgDispatcher):
    """DEPRECATED: use the MultiArgDispatcher instead!"""
