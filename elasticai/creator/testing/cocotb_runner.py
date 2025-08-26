from collections.abc import Callable, Iterable
from functools import partial
from os import environ
from pathlib import Path
from typing import Any

from cocotb.runner import get_runner  # type: ignore

from elasticai.creator.file_generation import find_project_root


def run_cocotb_sim_for_src_dir(
    src_files: Iterable[str] | Iterable[Path],
    top_module_name: str,
    cocotb_test_module: str,
    path2src: str = "",
    defines: dict | Callable[[], dict] = lambda: {},
    params: dict | Callable[[], dict] = lambda: {},
    timescale: tuple[str, str] = ("1ps", "1fs"),
    en_debug_mode: bool = True,
    waveform_save_dst: str = "",
) -> Path:
    """Function for running Verilog/VHDL Simulation using COCOTB environment
    :param src_files:           List with source files of each used Verilog/VHDL file
    :param top_module_name:     Name of the top module (from file)
    :param cocotb_test_module:  Fully qualified name of python module containing cotName of the cocotb testbench in Python
    :param path2src:            Path to the folder in which all src files are available for testing
    :param defines:             Dictionary of parameters to pass to the module [key: value, ...] - usable only in Verilog
    :param params:              Dictionary of parameters to pass to the module [key: value, ...] - value will be ignored
    :param timescale:           Tuple with Timescale value for simulation (step, accuracy)
    :param en_debug_mode:       Enable debug mode
    :param waveform_save_dst:   Path to the destination folder for saving waveform file
    :return:                    Path to folder which includes waveform file [Default: simulation output folder]
    """
    _path2src = Path(path2src)
    return run_cocotb_sim(
        src_files=[_path2src / f for f in src_files],
        top_module_name=top_module_name,
        cocotb_test_module=cocotb_test_module,
        defines=defines,
        params=params,
        timescale=timescale,
        en_debug_mode=en_debug_mode,
        waveform_save_dst=waveform_save_dst,
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
    defines: dict[str, Any] | Callable[[], dict[str, Any]] = lambda: {},
    params: dict[str, Any] | Callable[[], dict[str, Any]] = lambda: {},
    timescale: tuple[str, str] = ("1ps", "1fs"),
    en_debug_mode: bool = True,
    waveform_save_dst: str = "",
) -> Path:
    """Function for running Verilog/VHDL Simulation using COCOTB environment
    :param src_files:           List with source files of each used Verilog/VHDL file
    :param top_module_name:     Name of the top module (from file)
    :param cocotb_test_module:  Fully qualified name of python module containing cotName of the cocotb testbench in Python
    :param defines:             Dictionary of parameters to pass to the module [key: value, ...] - usable only in Verilog
    :param params:              Dictionary of parameters to pass to the module [key: value, ...] - value will be ignored
    :param timescale:           Tuple with Timescale value for simulation (step, accuracy)
    :param en_debug_mode:       Enable debug mode
    :param waveform_save_dst:   Path to the destination folder for saving waveform file
    :return:                    Path to folder which includes waveform file [Default: simulation output folder]
    """

    design_sources = [Path(m) for m in src_files]
    params = _normalize_dict_arg(params)
    defines = _normalize_dict_arg(defines)
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

    build_sim_dir = find_project_root() / "build_sim"
    build_sim_dir.mkdir(exist_ok=True, parents=True)
    build_waveform_dir = (
        build_sim_dir.absolute()
        if waveform_save_dst == ""
        else Path(waveform_save_dst).absolute()
    )
    build_waveform_dir.mkdir(exist_ok=True, parents=True)
    if language == "verilog":
        # --- Building new dump file for getting vcd file
        plus_args = []
        build_args = ["-s", "cocotb_iverilog_dump_v2"]
        dump_file_src = _create_iverilog_dump_file(
            top_module_name=top_module_name,
            dump_dst=build_sim_dir,
            path=build_waveform_dir,
        )
        design_sources.append(dump_file_src)
        build_call = partial(runner.build, verilog_sources=design_sources)
    else:
        top_module_name = top_module_name.lower()
        build_args = []
        plus_args = [f"--vcd={build_waveform_dir / 'waveforms'}.vcd"]
        build_call = partial(runner.build, vhdl_sources=design_sources)

    build_call(
        hdl_toplevel=top_module_name,
        always=True,
        clean=False,
        waves=True,
        build_args=build_args,
        defines=defines,
        parameters=params,
        timescale=timescale,
        build_dir=build_sim_dir,
    )

    runner.test(
        hdl_toplevel=top_module_name,
        test_module=[cocotb_test_module],
        hdl_toplevel_lang=language,
        gui=False,
        plusargs=plus_args,
        waves=True,
        parameters=params,
        timescale=timescale,
        build_dir=build_sim_dir,
        test_dir=build_sim_dir,
    )
    return build_waveform_dir.absolute()


def _create_iverilog_dump_file(
    top_module_name: str, path: Path, dump_dst: Path
) -> Path:
    dumpfile_path = path / "waveforms.vcd"
    with open(dump_dst / "dump.v", "w") as f:
        f.write("module cocotb_iverilog_dump_v2();\n")
        f.write("initial begin\n")
        f.write(f'    $dumpfile("{dumpfile_path}");\n')
        f.write(f"    $dumpvars(0, {top_module_name});\n")
        f.write("end\n")
        f.write("endmodule\n")
    return dump_dst / "dump.v"


def _normalize_dict_arg(
    arg: dict[str, Any] | Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if callable(arg):
        return arg()
    return arg
