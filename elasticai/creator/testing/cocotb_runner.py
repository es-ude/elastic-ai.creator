from collections.abc import Iterable
from functools import partial
from os import environ
from os.path import join
from pathlib import Path
from typing import Any

from cocotb.runner import get_runner

from elasticai.creator.file_generation import find_project_root


def get_and_create_sim_build_dir(folder_name: str) -> Path:
    build = find_project_root() / folder_name
    build.mkdir(exist_ok=True)
    return build


def run_cocotb_sim_for_src_dir(
    src_files: Iterable[str] | Iterable[Path],
    top_module_name: str,
    cocotb_test_module: str,
    path2src: str = "",
    defines: dict = {},
    params: dict = {},
    timescale: tuple[str, str] = ("1ps", "1fs"),
    en_debug_mode: bool = True,
    build_waveforms: bool = False,
) -> None:
    """Function for running Verilog/VHDL Simulation using COCOTB environment
    :param src_files:           List with source files of each used Verilog/VHDL file
    :param top_module_name:     Name of the top module (from file)
    :param cocotb_test_module:  Fully qualified name of python module containing cotName of the cocotb testbench in Python
    :param path2src:            Path to the folder in which all src files are available for testing
    :param defines:             Dictionary of parameters to pass to the module [key: value, ...] - usable only in Verilog
    :param params:              Dictionary of parameters to pass to the module [key: value, ...] - value will be ignored
    :param timescale:           Tuple with Timescale value for simulation (step, accuracy)
    :param en_debug_mode:       Enable debug mode
    :param build_waveforms:     Boolean for building the waveforms of the stimuli (will be saved in build_sim folder)
    :return:                    None
    """
    path2src = Path(path2src)
    return run_cocotb_sim(
        src_files=[path2src / f for f in src_files],
        top_module_name=top_module_name,
        cocotb_test_module=cocotb_test_module,
        defines=defines,
        params=params,
        timescale=timescale,
        en_debug_mode=en_debug_mode,
        build_waveforms=build_waveforms,
    )


def _build_language_map() -> dict[str, str]:
    language_mapping = {}
    for ks, language in [((".v", ".sv"), "verilog"), ((".vhd", ".vhdl"), "vhdl")]:
        for k in ks:
            language_mapping[k] = language
    return language_mapping


def run_cocotb_sim(
    src_files: Iterable[str] | Iterable[Path],
    top_module_name: str,
    cocotb_test_module: str,
    defines: dict[str, Any] = {},
    params: dict[str, Any] = {},
    timescale: tuple[str, str] = ("1ps", "1fs"),
    en_debug_mode: bool = True,
    build_waveforms: bool = False,
) -> None:
    """Function for running Verilog/VHDL Simulation using COCOTB environment
    :param src_files:           List with source files of each used Verilog/VHDL file
    :param top_module_name:     Name of the top module (from file)
    :param cocotb_test_module:  Fully qualified name of python module containing cotName of the cocotb testbench in Python
    :param defines:             Dictionary of parameters to pass to the module [key: value, ...] - usable only in Verilog
    :param params:              Dictionary of parameters to pass to the module [key: value, ...] - value will be ignored
    :param timescale:           Tuple with Timescale value for simulation (step, accuracy)
    :param en_debug_mode:       Enable debug mode
    :param build_waveforms:     Boolean for building the waveforms of the stimuli (will be saved in build_sim folder)
    :return:                    None
    """
    design_sources = [Path(m) for m in src_files]
    if len(design_sources) == 0:
        raise ValueError("no design sources specified")

    if any(map(lambda x: not x.exists(), design_sources)):
        raise FileNotFoundError(f"Design file does not exist {design_sources}")

    runner_mapping = {"verilog": "icarus", "vhdl": "ghdl"}
    suffix = design_sources[0].suffix
    language_mapping = _build_language_map()
    if suffix not in language_mapping:
        raise ValueError(f"File ending {suffix} not supported")

    language = language_mapping[suffix]
    runner = get_runner(runner_mapping[language])

    environ["COCOTB_RESOLVE_X"] = "ZEROS"
    environ["COCOTB_LOG_LEVEL"] = "INFO" if en_debug_mode else "WARNING"
    environ["COCOTB_REDUCED_LOG_FMT"] = "0" if en_debug_mode else "1"
    environ["MACOSX_DEPLOYMENT_TARGET"] = "15.0"

    if language == "verilog":
        build_call = partial(runner.build, verilog_sources=design_sources)
        plus_args = []
    else:
        top_module_name = top_module_name.lower()
        plus_args = [f"--vcd={top_module_name}.vcd"] if build_waveforms else []
        build_call = partial(runner.build, vhdl_sources=design_sources)

    build_call(
        hdl_toplevel=top_module_name,
        always=True,
        clean=True,
        waves=build_waveforms,
        defines=defines,
        parameters=params,
        timescale=timescale,
        build_dir=find_project_root() / "build_sim",
    )
    runner.test(
        hdl_toplevel=top_module_name,
        test_module=[cocotb_test_module],
        hdl_toplevel_lang=language,
        gui=False,
        plusargs=plus_args,
        waves=build_waveforms,
        parameters=params,
        timescale=timescale,
        build_dir=join(find_project_root(), "build_sim"),
        test_dir=join(find_project_root(), "build_sim"),
    )
