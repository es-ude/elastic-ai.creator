from re import Match

from .template import (
    AnalysingTemplateParameterType,
    Template,
    TemplateBuilder,
    TemplateParameterType,
)


class EntityTemplateParameter(AnalysingTemplateParameterType):
    """Find the entity in vhdl code and make its name available as a template paramter.

    Assumes that there is only one entity in the provided prototype.

    NOTE: Will only replace occurences where `name` is followed and preceded by a non-word character, ie.
    for an entity named `'skeleton'` the occurence `'architecture rtl of skeleton is'` will be
    replaced by `'architecture rtl of $entity is'` but the occurences `'skeleton_pkg'`,
    `'skeleton;'`, `'skeleton-'` or `skeleton.` will remain unaltered.
    IMPORTANT: This detail is likely to change in the future.
    """

    def __init__(self):
        self.analyse_regex = (
            r"(?i:(?<=entity )(?P<{value}>[a-zA-Z][a-zA-Z0-9_]*)(?=\s+is))"
        )
        self.regex = "<none>"

    def analyse(self, m: Match) -> None:
        if m.lastgroup is None:
            raise ValueError()
        original_name = m.group(m.lastgroup)
        self.regex = r"(?P<{{value}}>\b{}\b)".format(original_name)

    def replace(self, m: Match):
        return "$entity"


class ValueTemplateParameter(TemplateParameterType):
    """Find a value definition and make it settable via a template parameter.

    Searches for vhdl value definitions of the form `identifier : type`
    or `identifier : type :=` and transforms them into `identifier : type := $identifier`.

    Essentially allows to replace generics as well as variable or signal initializations.
    """

    def __init__(self):
        self.regex = r"(?i:{value}\s*:\s*(natural|integer))(?P<{value}>\b)"

    def replace(self, m: Match) -> str:
        return f"{m.group(0)} := ${m.lastgroup}"


class EntityTemplateDirector:
    def __init__(self):
        self._builder = TemplateBuilder()
        self._builder.add_parameter("entity", EntityTemplateParameter())

    def set_prototype(self, prototype: str) -> "EntityTemplateDirector":
        self._builder.set_prototype(prototype)
        return self

    def add_generic(self, name: str) -> "EntityTemplateDirector":
        if name == "entity":
            raise ValueError("name 'entity' is reserverd for entity parameter")
        self._builder.add_parameter(name, ValueTemplateParameter())
        return self

    def build(self) -> Template:
        return self._builder.build()
