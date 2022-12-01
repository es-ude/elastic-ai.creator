import os
from typing import Any, Iterable, Iterator

import torch

from elasticai.creator.resource_utils import PathType
from elasticai.creator.vhdl.components.network_component import NetworkVHDLFile
from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch.build_function_mappings import (
    DEFAULT_BUILD_FUNCTION_MAPPING,
)
from elasticai.creator.vhdl.vhdl_files import VHDLModule, VHDLBaseFile, VHDLBaseModule


def translate_model(
    model: torch.nn.Module,
    translation_args: dict[str, Any],
    build_function_mapping: BuildFunctionMapping = DEFAULT_BUILD_FUNCTION_MAPPING,
) -> Iterator[VHDLModule]:
    """
    Translates a given PyTorch-model to an intermediate representation. The intermediate representation is represented
    as an iterator of VHDLModule objects.

    Parameters:
        model (torch.nn.Module): The PyTorch-model that should be translated.
        translation_args (dict[str, Any]):
            Dictionary with the translation arguments for each kind of VHDLModule included in the vhdl_modules.
        build_function_mapping (BuildFunctionMapping):
            Object that maps a given PyTorch-layer to its corresponding build function. If not given the default build
            functions will be used.

    Returns:
        Iterator[CodeModule]:
            Iterator of containers that holds the actual VHDL code. The CodeModule objects holds all CodeFile
            objects for one VHDLModule object (module) and the name of that module. One CodeFile object
            holds the file name of that code files and the actual code as an iterable of str (lines).
    """
    flat_model = filter(
        lambda x: not isinstance(x, (type(model), torch.nn.Sequential)), model.modules()
    )

    for layer_index, layer in enumerate(flat_model):
        layer_class_name = type(layer).__name__

        build_fn = build_function_mapping.get_from_layer(layer)
        if build_fn is None:
            raise NotImplementedError(
                f"Layer '{layer_class_name}' is currently not supported."
            )
        module = build_fn(layer, str(layer_index))

        args = translation_args.get(layer_class_name)
        components = module.files
        files = map(lambda x: VHDLBaseFile(name=x.name, code=x.code), components)

        yield VHDLBaseModule(name=f"{layer_index}_{layer_class_name}", files=files)
    network = NetworkVHDLFile()
    network_file = VHDLBaseFile(name=network.name, code=network.code)
    yield VHDLBaseModule(name="network_component", files=[network_file])


def save_code(code_repr: Iterable[VHDLModule], path: PathType) -> None:
    """
    Saves the generated code on the file system.

    Parameters:
        code_repr (Iterable[VHDLModule]): The generated code that should be saved.
        path (PathType):
            The path to a folder in which the code should be saved. All parent folders that don't exist will be created.
    """
    os.makedirs(path)

    for module in code_repr:
        module_path = os.path.join(path, module.name)
        os.makedirs(module_path)

        for code_file in module.files:
            file_path = os.path.join(module_path, code_file.name)
            code = "\n".join(code_file.code)

            with open(file_path, "w") as out_file:
                out_file.write(code)
