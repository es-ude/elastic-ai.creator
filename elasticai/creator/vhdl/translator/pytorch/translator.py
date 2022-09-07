import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch

from elasticai.creator.resource_utils import PathType
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch.build_function_mappings import (
    DEFAULT_BUILD_FUNCTION_MAPPING,
)
from elasticai.creator.vhdl.vhdl_component import VHDLModule


@dataclass
class CodeFile:
    file_name: str
    code: Code


@dataclass
class CodeModule:
    module_name: str
    files: Iterable[CodeFile]


def translate_model(
    model: torch.nn.Module,
    build_function_mapping: BuildFunctionMapping = DEFAULT_BUILD_FUNCTION_MAPPING,
) -> Iterator[VHDLModule]:
    """
    Translates a given PyTorch-model to an intermediate representation. The intermediate representation is represented
    as an iterator of VHDLModule objects.

    Parameters:
        model (torch.nn.Module): The PyTorch-model that should be translated.
        build_function_mapping (BuildFunctionMapping):
            Object that maps a given PyTorch-layer to its corresponding build function. If not given the default build
            functions will be used.

    Returns:
        Iterator[VHDLModule]: Iterator that yields for each layer one VHDLModule object.
    """
    flat_model = filter(
        lambda x: not isinstance(x, torch.nn.Sequential), model.modules()
    )
    for layer in flat_model:
        build_fn = build_function_mapping.get_from_layer(layer)
        if build_fn is not None:
            yield build_fn(layer)


def generate_code(
    vhdl_modules: Iterable[VHDLModule],
    translation_args: dict[str, Any],
) -> Iterator[CodeModule]:
    """
    Takes an iterable of VHDLModule objects a dictionary of arguments for each VHDLModule object type and performs
    the translation from the intermediate representation to the actual VHDL code.

    Parameters:
        vhdl_modules (Iterable[VHDLModule]):
            The intermediate representation as an iterator of VHDLModule objects.
        translation_args (dict[str, dict[str, Any]]):
            Dictionary with the translation arguments for each kind of VHDLModule included in the vhdl_modules.

    Returns:
        Iterator[CodeModule]:
            Iterator of containers that holds the actual VHDL code. The CodeModule objects holds all CodeFile
            objects for one VHDLModule object (module) and the name of that module. One CodeFile object
            holds the file name of that code files and the actual code as an iterable of str (lines).
    """
    for module_index, module in enumerate(vhdl_modules):
        module_class_name = type(module).__name__

        args = translation_args.get(module_class_name)

        components = module.components(args)
        files = map(lambda x: CodeFile(file_name=x.file_name, code=x()), components)

        yield CodeModule(module_name=f"{module_index}_{module_class_name}", files=files)


def save_code(code_repr: Iterable[CodeModule], path: PathType) -> None:
    """
    Saves the generated code on the file system.

    Parameters:
        code_repr (Iterable[CodeModule]): The generated code that should be saved.
        path (PathType):
            The path to a folder in which the code should be saved. All parent folders that don't exist will be created.
    """
    for module in code_repr:
        module_path = os.path.join(path, module.module_name)
        os.makedirs(module_path, exist_ok=True)

        for code_file in module.files:
            file_path = os.path.join(module_path, code_file.file_name)
            code = "\n".join(code_file.code)

            with open(file_path, "w") as out_file:
                out_file.write(code)
