from dataclasses import dataclass
from inspect import signature as _signature


@dataclass
class PluginSpec:
    """The specification of a plugin.

    Typically built by reading a dictionary from a toml file and
    building the spec using <<build_plugin_spec, `build_plugin_spec()`>>.
    The dataclass is only used to provide convenient access to the fields,
    support type checking and improve code readability.

    You can achieve your goals just as well with the `PluginDict` dictionary.
    That is defined as an alias for `dict[str, str | tuple[str, ...]]`.
    """

    name: str
    target_platform: str
    target_runtime: str
    version: str
    api_version: str
    package: str


type PluginMap = dict[str, str | list[str] | PluginMap]


def build_plugin_spec[SpecT: PluginSpec](d: PluginMap, spec_type: type[SpecT]) -> SpecT:
    """inspect spec_type and build an instance of it from the dictionary `d`.

    Missing field raise an error while extra fields will be ignored.
    """
    args = d
    s = _signature(spec_type)
    expected_params = set(s.parameters.keys())
    actual_params = set(args.keys())
    if expected_params != actual_params:
        if not actual_params.issuperset(expected_params):
            raise MissingFieldError(
                expected_params.difference(actual_params),
                spec_type,
            )
    bound = {k: args[k] for k in s.parameters.keys()}
    return spec_type(**bound)  # type: ignore


class MissingFieldError[PluginSpecT: PluginSpec](Exception):
    def __init__(self, field_names: set[str], plugin_type: type[PluginSpecT]):
        super().__init__(
            f"missing required fields {field_names} for plugin spec '{plugin_type.__qualname__}'\n\tAre you sure you are loading the correct plugin?\n\tIs the meta.toml file correct?"
        )
