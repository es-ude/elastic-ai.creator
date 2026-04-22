import types
from collections.abc import Callable, Hashable, Mapping
from functools import update_wrapper, wraps
from typing import Concatenate, Protocol, Self, cast, overload


class Registrar[K: Hashable, **V, R](Protocol):
    """A decorator to register values. Keys are derived from values or provided explicitly."""

    @overload
    def __call__(self, key: K | None = None, /) -> Callable[V, R]: ...

    @overload
    def __call__(self, key: K, *args: V.args, **kwargs: V.kwargs) -> R: ...


def registrar_method[Owner, K: Hashable, **V, R](
    wrapped: Callable[Concatenate[Owner, K | None, V], R],
) -> Registrar[K, V, R]:
    """Turn a method into a registrar."""

    @wraps(wrapped)
    def wrapper(
        self: Owner, key: K | None = None, *args: V.args, **kwargs: V.kwargs
    ) -> R | Callable[V, R]:
        if len(args) + len(kwargs) == 0:

            def decorator(*args: V.args, **kwargs: V.kwargs) -> R:
                return wrapped(self, key, *args, **kwargs)

            return decorator

        else:
            return wrapped(self, key, *args, **kwargs)
        raise ValueError("Invalid arguments to TypeHandlerDecorator.")

    return cast(Registrar[K, V, R], wrapper)


def registrar[K: Hashable, **V, R](
    wrapped: Callable[Concatenate[K | None, V], R],
) -> Registrar[K, V, R]:
    """Turn a function into a registrar."""

    @wraps(wrapped)
    def wrapper(
        key: K | None = None, *args: V.args, **kwargs: V.kwargs
    ) -> R | Callable[V, R]:
        if len(args) + len(kwargs) == 0:

            def decorator(*args: V.args, **kwargs: V.kwargs) -> R:
                return wrapped(key, *args, **kwargs)

            return decorator
        else:
            return wrapped(key, *args, **kwargs)
        raise ValueError("Invalid arguments to TypeHandlerDecorator.")

    return cast(Registrar[K, V, R], wrapper)


class KeyedDispatcher[**P1, **P2, R1, R2, K: Hashable](Protocol):
    """Automatically dispatch registered functions based on a key derived from the arguments."""

    def register(self, key: K, fn: Callable[P2, R1], /) -> Callable[P2, R1]: ...

    def override(self, key: K, fn: Callable[P2, R1], /) -> Callable[P2, R1]: ...

    @property
    def registry(self) -> Mapping[K, Callable[P2, R1]]: ...

    def __call__(self, *args: P1.args, **kwargs: P1.kwargs) -> R2: ...


class KeyedDispatcherWithRegistrars[**P1, **P2, R1, R2, K: Hashable](Protocol):
    """Automatically dispatch registered functions based on a key derived from the arguments."""

    @property
    def register(self) -> Registrar[K, [Callable[P2, R1]], Callable[P2, R1]]: ...
    @property
    def override(self) -> Registrar[K, [Callable[P2, R1]], Callable[P2, R1]]: ...

    @property
    def registry(self) -> Mapping[K, Callable[P2, R1]]: ...

    def __call__(self, *args: P1.args, **kwargs: P1.kwargs) -> R2: ...


def create_keyed_dispatch[**P1, **P2, R1, R2, K: Hashable](
    key_from_obj: Callable[P1, K], key_from_fn: Callable[[Callable[P2, R1]], K]
) -> Callable[
    [Callable[Concatenate[Callable[P2, R1], P1], R2]],
    KeyedDispatcherWithRegistrars[P1, P2, R1, R2, K],
]:
    """The created decorator allows to register functions that will be passed to the decorated function based on a key.

    This is useful to create functions that dispatch based on some property of the arguments. Additionally, it allows to override
    previously registered functions. The decorated function will receive the dispatched function as its first argument, thus
    allowing clients to decide how to call the dispatched function.
    """

    def keyed_dispatch(
        fn: Callable[Concatenate[Callable[P2, R1], P1], R2],
    ) -> KeyedDispatcherWithRegistrars[P1, P2, R1, R2, K]:
        registry: dict[K, Callable[P2, R1]] = {}

        def wrapper(*args: P1.args, **kwargs: P1.kwargs) -> R2:
            key = key_from_obj(*args, **kwargs)
            if key not in registry:
                raise KeyError(f"No function registered for key: {key}")
            handler_fn = registry[key]
            return fn(handler_fn, *args, **kwargs)

        @registrar
        def register(key: K | None, fn: Callable[P2, R1]) -> Callable[P2, R1]:
            if key is None:
                key = key_from_fn(fn)
            if key in registry:
                raise ValueError(f"Function already registered for key: {key}")
            registry[key] = fn
            return fn

        @registrar
        def override(key: K | None, fn: Callable[P2, R1]) -> Callable[P2, R1]:
            if key is None:
                key = key_from_fn(fn)
            if key not in registry:
                raise ValueError(f"No function registered for key: {key} to override")
            registry[key] = fn
            return fn

        setattr(wrapper, "register", register)
        setattr(wrapper, "override", override)
        setattr(wrapper, "registry", types.MappingProxyType(registry))

        update_wrapper(wrapper, fn)

        return cast(KeyedDispatcherWithRegistrars[P1, P2, R1, R2, K], wrapper)

    return keyed_dispatch


class KeyedDispatcherForDescriptor[**P1, **P2, R1, R2, Owner, K: Hashable](
    KeyedDispatcher[P1, P2, R1, R2, K]
):
    __slots__ = ("_owner", "_key_from_obj", "_registry_name", "_fn")

    def __init__(
        self,
        owner: Owner,
        registry_name: str,
        key_from_obj: str,
        fn: Callable[Concatenate[Owner, Callable[P2, R1], P1], R2],
    ) -> None:
        self._owner = owner
        self._key_from_obj = key_from_obj
        self._registry_name = registry_name
        self._fn = fn

    def _registry_dict(self) -> dict[K, Callable[P2, R1]]:
        if not hasattr(self._owner, self._registry_name):
            setattr(self._owner, self._registry_name, {})
        reg = getattr(self._owner, self._registry_name)
        return reg

    @property
    def registry(self) -> Mapping[K, Callable[P2, R1]]:
        return types.MappingProxyType(self._registry_dict())

    def register(self, key: K, fn: Callable[P2, R1]) -> Callable[P2, R1]:
        if key in self.registry:
            raise ValueError(f"Function already registered for key: {key}")
        self._registry_dict()[key] = fn
        return fn

    def override(self, key: K, fn: Callable[P2, R1]) -> Callable[P2, R1]:
        if key not in self.registry:
            raise ValueError(f"No function registered for key: {key} to override")
        self._registry_dict()[key] = fn
        return fn

    def _get_key_from_args(self, *args: P1.args, **kwargs: P1.kwargs) -> K:
        key_fn = getattr(self._owner, self._key_from_obj)
        return key_fn(*args, **kwargs)

    def __call__(self, *args: P1.args, **kwargs: P1.kwargs) -> R2:
        key = self._get_key_from_args(*args, **kwargs)
        registry = self._registry_dict()
        if key not in registry:
            raise KeyError(f"No function registered for key: {key}")
        handler_fn = registry[key]

        def bound_call(
            handler_fn: Callable[P2, R1],
            *args: P1.args,
            **kwargs: P1.kwargs,
        ) -> R2:
            return self._fn(self._owner, handler_fn, *args, **kwargs)

        return bound_call(handler_fn, *args, **kwargs)


class KeyedDispatcherDescriptor[**P1, **P2, R1, R2, Owner, K: Hashable]:
    """Automatically dispatch registered functions based on a key derived from the arguments.

    Opposed to its function counterpart, this descriptor is intended to be used as a class attribute.

    Use the decorators as follows

     - `key_from_args` on a method of the owning class to define how to derive the key from the arguments
     - `dispatch_for` on a method of the owning class to define the dispatching behavior. The method will receive the dispatched function as its first argument followed by the original arguments.
    """

    __slots__ = ("_key_from_obj", "_registry", "_name", "_fn")

    def __init__(
        self, fn: Callable[Concatenate[Owner, Callable[P2, R1], P1], R2] | None = None
    ) -> None:
        self._fn: Callable[Concatenate[Owner, Callable[P2, R1], P1], R2] | None = fn
        self._key_from_obj: str = ""
        self._registry: str = ""

    def __get__(
        self, instance: Owner | None, owner: type[Owner]
    ) -> KeyedDispatcher[P1, P2, R1, R2, K]:
        if instance is None:
            raise TypeError(
                "KeyedDispatcherDescriptor must be accessed via an instance."
            )
        if self._key_from_obj == "":
            raise TypeError("Key function not set. Use the key_from_args decorator.")
        if self._fn is None:
            raise TypeError("specified dispatch function was None")
        return KeyedDispatcherForDescriptor(
            instance, self._registry, self._key_from_obj, self._fn
        )

    def key_from_args(
        self, key_from_obj: Callable[Concatenate[Owner, P1], K]
    ) -> Callable[Concatenate[Owner, P1], K]:
        if hasattr(key_from_obj, "__name__") and isinstance(key_from_obj.__name__, str):
            self._key_from_obj = key_from_obj.__name__
            return key_from_obj
        else:
            raise Exception("Use @dispatcher.key_from_args on a method object")

    def __set_name__(self, instance: Owner, name: str) -> None:
        self._registry = f"_{name}_registry"

    def _unbound_call(
        self,
        owner: Owner,
        *args: P1.args,
        **kwargs: P1.kwargs,
    ) -> R2:
        if self._fn is None:
            raise TypeError(
                "Dispatch function not set. Use the dispatch_for decorator."
            )
        dispatcher: KeyedDispatcher[P1, P2, R1, R2, K] = KeyedDispatcherForDescriptor(
            owner, self._registry, self._key_from_obj, self._fn
        )
        return dispatcher(*args, **kwargs)

    def dispatch_for(
        self,
        fn: Callable[Concatenate[Owner, Callable[P2, R1], P1], R2],
    ) -> Self:
        self._fn = fn
        return self


def dispatch_method[**P1, **P2, R1, R2, Owner, K: Hashable](
    _: type[K],
) -> Callable[
    [Callable[Concatenate[Owner, Callable[P2, R1], P1], R2]],
    KeyedDispatcherDescriptor[P1, P2, R1, R2, Owner, K],
]:
    """A decorator to create a KeyedDispatcherDescriptor for methods.

    This often has the advantage of not needing to specify the generic
    parameters explicitly. However, the type of keys cannot be inferred
    automatically, this is why it needs to be provided as an argument to
    the decorator.

    Note: The ParamSpec `**P` will narrow down to the most concrete type.
    Thus if you decorate a method with a signature like
    ```python
    @dispatch_method(str)
    def call(self, fn: Callable[[str], str], item: str) -> str:
        ...
    ```
    The type checker will assume that `P` is `(item: str)`.
    This means that annotating another method that should
    be compatible with this dispatcher might (rightly) lead to type errors.
    A registration method could e.g., look like this:
    ```python
    def register(self, name: str | None, fn: Callable[[str], str]) -> Callable[[str], str]:
        if name is None:
            name = fn.__name__
        self.call.register(name, fn)
        return fn
    ```
    However, this will not pass type checking, as the checker expects
    the signature to be `Callable[[Arg('item', str)], str]`.
    Sadly there is no way to express this inline in current python versions.
    You have the following options
        - make the arguments more generic (e.g., using `Any`) and loose type safety
        - use positional only arguments in the `call` method like so:
          ```python
          @dispatch_method(str)
          def call(self, fn: Callable[[str], str], item: str, /) -> str:
                ...
          ```
        - explicitly annotate the descriptor attribute in the class like so:
          ```python
          class MyClass:
              call: KeyedDispatcherDescriptor[[str], str, str, "MyClass", str] = KeyedDispatcherDescriptor()


    Using a protocol to define the expected signature of the registered functions will not help here, as the type checker will not be able
    to map the protocol to the ParamSpec `P` and return value `R1`.
    """

    def decorator(
        wrapped: Callable[Concatenate[Owner, Callable[P2, R1], P1], R2],
    ) -> KeyedDispatcherDescriptor[P1, P2, R1, R2, Owner, K]:
        return KeyedDispatcherDescriptor(wrapped)

    return decorator
