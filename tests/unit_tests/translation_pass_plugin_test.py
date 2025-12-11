from collections.abc import Callable, Iterable

import elasticai.creator.function_dispatch as F
import elasticai.creator.translation_pass_plugin as tpp


def test_can_use_type_handler_to_provide_loadable_symbols() -> None:
    def _get_key_from_obj(item: str) -> str:
        return item

    def _get_key_from_fn(fn: Callable[[str], str]) -> str:
        return fn.__name__

    @F.create_keyed_dispatch(_get_key_from_obj, _get_key_from_fn)
    def translate(fn: Callable[[str], str], item: str) -> str:
        return f"translated {fn(item)}"

    @tpp.type_handler()
    def a(t) -> str:
        return "ta"

    @tpp.type_handler("b")
    def handle_b(t) -> str:
        return "tb"

    a.load_into(translate)
    handle_b.load_into(translate)
    result = list(map(translate, ["a", "b"]))
    assert result == ["translated ta", "translated tb"]


def test_can_use_loadable_symbols_with_class_based_translation_pass() -> None:
    class MyTranslationPass:
        @F.dispatch_method(str)
        def _process_item(self, fn: Callable[[str], str], item: str, /) -> str:
            return f"translated {fn(item)}"

        def __call__(self, *items: str) -> Iterable[str]:
            for item in items:
                yield self._process_item(item)

        @_process_item.key_from_args
        def _key_from_item(self, item: str) -> str:
            return item

        @F.registrar_method
        def override(
            self, name: str | None, fn: Callable[[str], str]
        ) -> Callable[[str], str]:
            if name is None:
                name = fn.__name__
            self._process_item.override(name, fn)
            return fn

        @F.registrar_method
        def register(
            self, name: str | None, fn: Callable[[str], str]
        ) -> Callable[[str], str]:
            if name is None:
                name = fn.__name__
            self._process_item.register(name, fn)
            return fn

    translate = MyTranslationPass()

    @tpp.type_handler()
    def a(t) -> str:
        return "ta"

    @tpp.type_handler("b")
    def handle_b(t) -> str:
        return "tb"

    a.load_into(translate)
    handle_b.load_into(translate)

    assert ["translated ta", "translated tb"] == list(translate("a", "b"))
