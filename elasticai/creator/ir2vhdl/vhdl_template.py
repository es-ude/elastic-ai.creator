from string import Template as _pyTemplate

from elasticai.creator.template import (
    AnalysingTemplateParameter,
    TemplateBuilder,
    TemplateParameter,
)


class EntityTemplateParameter(AnalysingTemplateParameter):
    """Find the entity in vhdl code and make its name available as a template paramter.

    Assumes that there is only one entity in the provided prototype.

    NOTE: Will only replace occurences where `name` is followed and preceded by a non-word character, ie.
    for an entity named `'skeleton'` the occurence `'architecture rtl of skeleton is'` will be
    replaced by `'architecture rtl of $entity is'` but the occurences `'skeleton_pkg'`,
    `'skeleton;'`, `'skeleton-'` or `skeleton.` will remain unaltered.
    IMPORTANT: This detail is likely to change in the future.
    """

    def __init__(self):
        self.name = "entity"
        self.analyse_regex = (
            r"(?i:(?<=entity )(?P<param>[a-zA-Z][a-zA-Z0-9_]*)(?=\s+is))"
        )
        self.regex = "<none>"

    def analyse(self, m: dict[str, str]) -> None:
        original_name = m["param"]
        self.regex = r"\b{original_name}\b".format(original_name=original_name)

    def replace(self, m: dict[str, str]) -> str:
        return "$entity"


class ValueTemplateParameter(TemplateParameter):
    """Find a value definition and make it settable via a template parameter.

    Searches for vhdl value definitions of the form `identifier : type`
    or `identifier : type :=` and transforms them into `identifier : type := $identifier`.

    Essentially allows to replace generics as well as variable or signal initializations.
    """

    def __init__(self, name: str):
        self.regex = (
            r"(?P<def>(?i:{name}\s*:\s*(natural|integer)))\s*(:=\s*.*)?\b".format(
                name=name
            )
        )
        self.name = name

    def replace(self, match: dict[str, str]) -> str:
        return f"{match['def']} := ${self.name}"


class EntityTemplateDirector:
    def __init__(self):
        self._builder = TemplateBuilder()
        self._builder.add_parameter(EntityTemplateParameter())

    def set_prototype(self, prototype: str) -> "EntityTemplateDirector":
        self._builder.set_prototype(prototype)
        return self

    def add_generic(self, name: str) -> "EntityTemplateDirector":
        if name == "entity":
            raise ValueError("name 'entity' is reserved for entity parameter")
        self._builder.add_parameter(ValueTemplateParameter(name))
        return self

    def build(self) -> _pyTemplate:
        return _pyTemplate(self._builder.build())
