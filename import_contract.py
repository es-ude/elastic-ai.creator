from abc import ABC
from functools import partial
from typing import Any, Callable, Iterable, cast

from importlinter import Contract, ContractCheck, ImportGraph, fields, output
from importlinter.application.output import ERROR


class Module:
    def __init__(self, name: str, graph: ImportGraph):
        self.name = name
        self.graph = graph

    def __str__(self):
        return f"Module({self.name})"

    def __repr__(self):
        return str(self)

    @property
    def parent(self) -> "Module":
        return Module(".".join(self.name.split(".")[:-1]), self.graph)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Module):
            return False
        return self.graph == other.graph and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.graph))

    def is_sibling_of(self, other: "Module") -> bool:
        return self.parent == other.parent

    def _build_module(self, name: str) -> "Module":
        return Module(name=name, graph=self.graph)

    def descendants(self) -> Iterable["Module"]:
        return map(self._build_module, self.graph.find_descendants(self.name))

    def is_descendant_of(self, other: "Module") -> bool:
        return self.name.startswith(other.name)

    def is_child_of(self, other: "Module") -> bool:
        return other == self.parent

    def is_in_superpackage_of(self, other: "Module") -> bool:
        return other in self.parent.descendants()

    def children(self) -> Iterable["Module"]:
        return map(self._build_module, self.graph.find_children(self.name))

    def directly_imported(self) -> Iterable["Module"]:
        return map(
            self._build_module, self.graph.find_modules_directly_imported_by(self.name)
        )

    def is_private(self) -> bool:
        return self.name.split(".")[-1].startswith("_")

    def directly_imports(self, other: "Module") -> bool:
        return other.name in self.graph.find_modules_directly_imported_by(self.name)


class _BaseContract(Contract, ABC):
    @staticmethod
    def _fields_to_modules(fields, graph) -> set[Module]:
        make_module: Callable[[str], Module] = partial(Module, graph=graph)
        return set(map(make_module, map(str, cast(Iterable, fields))))

    def render_broken_contract(self, check: "ContractCheck") -> None:
        violations: dict[str, list] = check.metadata
        output.print_error(f"contract violations in {len(violations)} modules")
        for importer, details in violations.items():
            output.new_line()
            output.print_heading(level=3, text=importer, style=ERROR)
            for detail in details:
                line_numer = detail["line_number"]
                content = detail["line_contents"]
                output.print_error(f"\t{importer}: {line_numer}: {content}")

    def _collect_metadata(
        self, violations: list[tuple[str, str]], graph: ImportGraph
    ) -> dict[str, list]:
        metadata: dict[str, list] = {}
        for importer, imported in violations:
            if importer not in metadata:
                metadata[importer] = []
            metadata[importer].extend(
                graph.get_import_details(importer=importer, imported=imported)
            )
        return metadata


class PackagePrivateModules(_BaseContract):
    """
    Ensure that private modules (marked by leading underscore) are only
    imported sibling modules or modules in subpackagespoetry
    """

    def check(self, graph: ImportGraph, verbose: bool) -> "ContractCheck":
        violations = []
        targets = self._fields_to_modules(self.targets, graph)
        sources = self._fields_to_modules(self.sources, graph)
        sources = {child for source in sources for child in source.descendants()}
        output.print_warning(str(targets))

        def _module_is_target(module: Module) -> bool:
            return any(map(module.is_descendant_of, targets))

        output.print_warning(str(list(map(_module_is_target, targets))))

        for importer in sources:
            for imported in importer.directly_imported():
                if (
                    _module_is_target(imported)
                    and imported.is_private()
                    and not imported.is_sibling_of(importer)
                    and not imported.is_child_of(importer)
                    and not imported.is_in_superpackage_of(importer)
                ):
                    violations.append((importer.name, imported.name))
        return ContractCheck(
            kept=len(violations) == 0,
            metadata=self._collect_metadata(violations, graph),
        )

    sources = fields.SetField(fields.StringField())
    targets = fields.SetField(fields.StringField())
