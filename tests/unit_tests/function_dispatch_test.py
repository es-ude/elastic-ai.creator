from collections.abc import Callable, Iterator
from typing import Protocol, cast, overload

import pytest

from elasticai.creator.function_dispatch import (
    KeyedDispatcherDescriptor,
    KeyedDispatcherWithRegistrars,
    create_keyed_dispatch,
    dispatch_method,
    registrar_method,
)


def test_can_register_new_functions():
    @create_keyed_dispatch(lambda x: x, lambda fn: fn.__name__)
    def translate(fn: Callable[[str], str], x: str) -> str:
        return f"string: {fn(x)}"

    @translate.register()
    def a(x: str) -> str:
        return "a"

    @translate.register()
    def b(x: str) -> str:
        return "b"

    print(translate.registry)

    assert translate("a") == "string: a"


class Dispatcher(
    KeyedDispatcherWithRegistrars[[str], [str], str, Iterator[str], str], Protocol
):
    def run(self, *names: str) -> Iterator[str]: ...


def dict_based_registry() -> Dispatcher:
    class MyClass:
        def __init__(self):
            self._registry = {}

        @registrar_method
        def register(
            self, name: str | None, fn: Callable[[str], str]
        ) -> Callable[[str], str]:
            if name is None:
                name = fn.__name__  # type: ignore
            self._registry[name] = fn
            return fn

        def __call__(self, name: str) -> str:
            fn = self._registry[name]
            return fn(name)

        def run(self, *names: str) -> Iterator[str]:
            for name in names:
                yield self(name)

    return cast(Dispatcher, MyClass())


def descriptor_based_registry() -> Dispatcher:
    class MyClass:
        _registry: KeyedDispatcherDescriptor[[str], [str], str, str, "MyClass", str] = (
            KeyedDispatcherDescriptor()
        )

        @_registry.key_from_args
        def _get_key_from_args(self, arg: str) -> str:
            return arg

        @overload
        def register(
            self, name: str | None = None, /
        ) -> Callable[[Callable[[str], str]], Callable[[str], str]]: ...

        @overload
        def register(
            self, name: str, fn: Callable[[str], str], /
        ) -> Callable[[str], str]: ...

        def register(
            self, name: str | None = None, fn: Callable[[str], str] | None = None, /
        ) -> (
            Callable[[str], str]
            | Callable[[Callable[[str], str]], Callable[[str], str]]
        ):
            match (name, fn):
                case None, None:

                    def decorator(
                        fn: Callable[[str], str],
                    ) -> Callable[[str], str]:
                        return self._registry.register(fn.__name__, fn)  # type: ignore

                    return decorator

                case _, None:

                    def decorator(
                        fn: Callable[[str], str],
                    ) -> Callable[[str], str]:
                        return self._registry.register(name, fn)  # type: ignore

                    return decorator
                case None, _:
                    return self._registry.register(fn.__name__, fn)  # type: ignore
                case _, _:
                    return self._registry.register(name, fn)  # type: ignore
            raise ValueError("Invalid arguments to register.")

        @_registry.dispatch_for
        def __call__(self, fn: Callable[[str], str], name: str) -> str:
            return fn(name)

        def run(self, *names: str) -> Iterator[str]:
            for name in names:
                yield self(name)

    return cast(Dispatcher, MyClass())


def registrar_based_registry() -> Dispatcher:
    class MyClass:
        _registry: KeyedDispatcherDescriptor[[str], [str], str, str, "MyClass", str] = (
            KeyedDispatcherDescriptor()
        )

        @_registry.key_from_args
        def _get_key_from_args(self, arg: str) -> str:
            return arg

        @registrar_method
        def register(
            self, name: str | None, fn: Callable[[str], str], /
        ) -> Callable[[str], str]:
            if name is None:
                name = fn.__name__  # type: ignore

            return self._registry.register(name, fn)

        @_registry.dispatch_for
        def __call__(self, fn: Callable[[str], str], name: str) -> str:
            return fn(name)

        def run(self, *names: str) -> Iterator[str]:
            for name in names:
                yield self(name)

    return cast(Dispatcher, MyClass())


@pytest.mark.parametrize(
    "dispatcher",
    [dict_based_registry(), descriptor_based_registry(), registrar_based_registry()],
)
def test_check_registering_fns(dispatcher: Dispatcher) -> None:
    @dispatcher.register("bob")
    @dispatcher.register()
    def alice(name: str) -> str:
        return f"Hello, {name}!"

    assert list(dispatcher.run("alice", "bob")) == ["Hello, alice!", "Hello, bob!"]


def test_each_class_instance_has_its_own_registry() -> None:
    class MyClass:
        @dispatch_method(str)
        def do_call(self, fn: Callable[[str], str], name: str, /) -> str:
            return fn(name)

        def __call__(self, name: str) -> str:
            return self.do_call(name)

        @do_call.key_from_args
        def _get_key_from_args(self, arg: str) -> str:
            return arg

        @registrar_method
        def register(
            self, name: str | None, fn: Callable[[str], str]
        ) -> Callable[[str], str]:
            if name is None:
                name = fn.__name__  # type: ignore

            return self.do_call.register(name, fn)

    first = MyClass()
    second = MyClass()

    @first.register()
    @second.register()
    def alice(name: str) -> str:
        return f"Hello, {name}!"

    assert "Hello, alice!" == first("alice")
    assert "Hello, alice!" == second("alice")
