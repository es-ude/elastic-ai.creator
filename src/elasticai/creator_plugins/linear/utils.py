from pathlib import Path
from typing import Any

import elasticai.creator.ir2verilog as ir
from elasticai.creator.file_generation import find_project_root as get_path_to_build
from elasticai.creator.ir import Registry, attribute
from elasticai.creator.ir2verilog import Ir2Verilog, factory


def _load_and_plugin(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list,
    path2save: str | Path = get_path_to_build() / "build",
) -> None:
    def _build_verilog_implementation(
        type: str, id: str, params: dict[str, Any]
    ) -> ir.DataGraph:
        mod_name = f"{type}" + (f"_{id}" if id else "")
        return factory.graph(
            attributes=attribute(**params),
            type=type,
            name=mod_name.lower(),
        )

    def _prepare_translator(plugin_types: list[str]) -> Ir2Verilog:
        _translate = Ir2Verilog()
        loader = ir.PluginLoader(_translate)
        for plugin in plugin_types:
            loader.load_from_package(plugin)
        return _translate

    design = _build_verilog_implementation(type=type, id=id, params=params)

    build_dir = Path(f"{path2save}/")
    build_dir.mkdir(exist_ok=True)

    translate = _prepare_translator(packages)
    for name, content in translate(design, Registry()):
        (build_dir / name).write_text("".join(content))


def load_and_plugin(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list,
    path2save: str | Path = get_path_to_build() / "build",
) -> None:

    _load_and_plugin(
        type=type, id=id, params=params, packages=packages, path2save=path2save
    )
    _load_and_plugin(
        type="mac_array",
        id="",
        params={
            "BITWIDTH": params["BITWIDTH"],
            "NUM_MULT": params["NUM_MULT"],
            "SIZE_INPUT": params["SIZE_INPUT"],
            "BITS_SCALE_BIAS": params["BITS_SCALE_BIAS"],
            "BITS_SCALE_DOUT": params["BITS_SCALE_DOUT"],
        },
        packages=["mac"],
        path2save=path2save,
    )
    _load_and_plugin(
        type="mult_dsp_signed",
        id="",
        params={
            "BITWIDTH": params["BITWIDTH"],
        },
        packages=["multipliers"],
        path2save=path2save,
    )
