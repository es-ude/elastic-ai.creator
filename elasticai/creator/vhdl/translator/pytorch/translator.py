import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch.nn
from torch.nn import Module

from elasticai.creator.resource_utils import PathType
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.translator.abstract.translatable import Translatable
from elasticai.creator.vhdl.translator.pytorch.build_function_mapping import (
    BuildFunctionMapping,
)


@dataclass
class CodeFile:
    file_name: str
    code: Code


@dataclass
class ModuleDirectory:
    dir_name: str
    files: Iterable[CodeFile]


def translate_model(
    model: Module, build_function_mapping: BuildFunctionMapping
) -> Iterator[Translatable]:
    flat_model = filter(
        lambda x: not isinstance(x, torch.nn.Sequential), model.modules()
    )
    for layer in flat_model:
        build_fn = build_function_mapping.get(layer)
        if build_fn is not None:
            yield build_fn(layer)


def generate_code(
    translatable_layers: Iterable[Translatable],
    translation_args: dict[str, dict[str, Any]],
) -> Iterator[ModuleDirectory]:
    for module_index, module in enumerate(translatable_layers):
        module_class_name = type(module).__name__

        args = translation_args.get(module_class_name)
        args = dict() if args is None else args

        components = module.translate(**args)
        files = map(lambda x: CodeFile(file_name=x.file_name, code=x()), components)

        yield ModuleDirectory(
            dir_name=f"{module_index}_{module_class_name}", files=files
        )


def save_code(code: Iterable[ModuleDirectory], path: PathType) -> None:
    for module in code:
        module_path = os.path.join(path, module.dir_name)
        os.makedirs(module_path, exist_ok=True)

        for code_file in module.files:
            file_path = os.path.join(module_path, code_file.file_name)
            code = "\n".join(code_file.code)

            with open(file_path, "w") as out_file:
                out_file.write(code)
