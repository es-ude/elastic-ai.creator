from datetime import datetime
from pathlib import Path
from typing import Any

import elasticai.creator.ir2verilog as ir
from elasticai.creator.file_generation import find_project_root as get_path_to_build
from elasticai.creator.ir import Registry, attribute
from elasticai.creator.ir2verilog import Ir2Verilog, factory


def build_verilog_design(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list,
    build_tb: bool = False,
    path2save: str | Path = get_path_to_build() / "build",
) -> None:
    design = _build_verilog_implementation(
        type=type, id=id, params=params, build_tb=build_tb
    )

    build_dir = Path(f"{path2save}/")
    build_dir.mkdir(exist_ok=True)

    translate = _prepare_translator(packages)
    for name, content in translate(design, Registry()):
        (build_dir / name).write_text("".join(content))


def _build_verilog_implementation(
    type: str, id: str, params: dict[str, Any], build_tb: bool = False
) -> ir.DataGraph:
    mod_name = f"{type}_{id}"
    return factory.graph(
        attribute(
            build_tb=build_tb,
            date_copy_created=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            module_name=mod_name.upper(),
        )
        | params,
        type=type,
        name=mod_name.lower(),
    )


def _prepare_translator(plugin_types: list[str]) -> Ir2Verilog:
    """Function for entering the lowering pass of the Intermediate Representation to geht the design specification"""
    _translate = Ir2Verilog()
    loader = ir.PluginLoader(_translate)
    for plugin in plugin_types:
        loader.load_from_package(plugin)
    return _translate
