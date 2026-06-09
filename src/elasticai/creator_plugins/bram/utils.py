from pathlib import Path
from typing import Any

import elasticai.creator.ir2verilog as ir
from elasticai.creator.file_generation import find_project_root as get_path_to_build
from elasticai.creator.ir import Registry, attribute
from elasticai.creator.ir2verilog import Ir2Verilog, factory


def translate_path_to_int(path: Path) -> int:
    path_new = path.as_posix().encode("ascii")
    return int.from_bytes(path_new, byteorder="big") | (ord("/") << (len(path_new) * 8))


def write_mem_file(path: Path, data: list[int], bitwidth: int) -> None:
    hex_digits = (bitwidth + 3) // 4
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.as_posix(), "w") as f:
        for value in data:
            f.write(f"{value:0{hex_digits}x}\n")
        f.close()


def load_and_plugin(
    type: str,
    id: str,
    params: dict[str, Any],
    packages: list,
    path2save: str | Path = get_path_to_build() / "build",
) -> None:
    design = _build_verilog_implementation(type=type, id=id, params=params)

    build_dir = Path(f"{path2save}/")
    build_dir.mkdir(exist_ok=True)

    translate = _prepare_translator(packages)
    for name, content in translate(design, Registry()):
        (build_dir / name).write_text("".join(content))


def _build_verilog_implementation(
    type: str, id: str, params: dict[str, Any]
) -> ir.DataGraph:
    mod_name = f"{type}_{id}"
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
