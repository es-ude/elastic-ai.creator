from typing import Callable, Generic, TypeVar, overload

Tin = TypeVar("Tin")
Tout = TypeVar("Tout")
FN = TypeVar("FN", bound=Callable)


class FunctionDecoratorFactory(Generic[FN, Tout]):
    """Apply a callback to functions and either their own or custom names.

    IMPORTANT: if you want to use this as a decorator, do not forget to
    return the wrapped function from your callback.

    Parameters
    ----------
    callback : will be called as `callback(name, fn)`


    Example
    -------
    >>> registry = dict()
    >>> def register_fn(name, fn):
    ...  registry[name] = fn
    ...  return fn
    ...
    >>> register = FunctionDecoratorFactory(register_fn)
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
      return fn

    register = FunctionDecoratorFactory(register_fn)

    @register("other_name")
    @register
    def my_fn(x):
      print(x)
    ```

    This will add `my_fn` to the registry using the
    name `"my_fn"` and the name `"other_name"`.

    """

    def __init__(self, callback: Callable[[str, FN], Tout]):
        self._cb = callback

    @overload
    def __call__(self, name: str, /) -> Callable[[FN], Tout]:
        """Return a function that can be used to register another function with `name`."""
        ...

    @overload
    def __call__(self, fn: FN, /) -> Tout:
        """Register `fn` using its own name."""
        ...

    def __call__(self, arg: FN | str, /) -> Tout | Callable[[FN], Tout]:
        if isinstance(arg, str):
            return self.__reg_by_name(arg)
        return self.__reg(arg)

    def __reg_by_name(self, name: str) -> Callable[[FN], Tout]:
        def reg(fn: FN) -> FN:
            return self._cb(name, fn)

        return reg

    def __reg(self, fn: FN) -> Tout:
        return self._cb(fn.__name__, fn)


class RegisterDescriptor(Generic[Tin, Tout]):
    """Automatically connect the `FunctionDecoratorFactory` to a callback and make it look like a method.

    The owning instance needs to define a callback that has the name `f"_{name}_callback"`,
    where `name` is the name of the field assigned to this descriptor.

    For an example see the `MultiArgDispatcher` below.
    """

    def __set_name__(self, instance, name):
        self._cb_name = f"_{name}_callback"

    def __get__(
        self, instance, owner=None
    ) -> FunctionDecoratorFactory[Callable[[Tin], Tout], Callable[[Tin], Tout]]:
        cb = getattr(instance, self._cb_name)

        def wrapped(name, fn):
            cb(name, fn)
            return fn

        return FunctionDecoratorFactory(wrapped)


class KeyedFunctionDispatcher(Generic[Tin, Tout]):
    """
    Functions can be registered with custom key or by
    using the function's name as key.
    Upon `call` the registry will use the function
    specified by the `dispatch_key_fn` and the argument
    passed to `call` to process the argument and return the result.
    """

    register: RegisterDescriptor[Tin, Tout] = RegisterDescriptor()

    def __init__(self, dispatch_key_fn: Callable[[Tin], str]) -> None:
        self._key_fn = dispatch_key_fn
        self._fns: dict[str, Callable[[Tin], Tout]] = dict()

    def _register_callback(
        self, name: str, fn: Callable[[Tin], Tout]
    ) -> Callable[[Tin], Tout]:
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


class FunctionRegistry(KeyedFunctionDispatcher):
    """DEPRECATED: use the MultiArgDispatcher instead!"""
