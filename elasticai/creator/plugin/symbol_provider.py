from collections.abc import Callable
from functools import update_wrapper

from .symbol import PluginSymbolFn


def make_plugin_symbol[RecT, **P, ReturnT](
    load_into: Callable[[RecT], None], fn: Callable[P, ReturnT]
) -> PluginSymbolFn[RecT, P, ReturnT]:
    """Turn two functions into a loadable and callable plugin symbol.

    .Args
    * `load_into`: executed when `PluginLoader` calls `load_into` on the Symbol
    * `fn`: wrapped function. Executed when the created `PluginSymbolFn` is called.

    An important use case for plugin symbols is to provide callable functions,
    e.g., for lowering passes. `make_plugin_symbol` eases creating
    plugin symbols for these functions.

    For an example look at the implementation of the `type_handler`
    decorators in xref::api:ir2vhdl.adoc[ir2vhdl].
    ```
    """

    class _PS(PluginSymbolFn):
        def load_into(self, receiver):
            load_into(receiver)

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ReturnT:
            return fn(*args, **kwargs)

    wrapped = _PS()

    update_wrapper(wrapped, fn)
    return wrapped
