import json
from pathlib import Path

from elasticai.creator.file_generation import find_project_root


def build_report_folder_and_testpattern(
    dut_name: str, testpattern: dict, file_appendix: str = ""
) -> Path:
    """Building the test/simulation folder which contains the testpattern data and hardware design for testing
    :param dut_name:        The name of the Top Module
    :param testpattern:     Dictionary with testpattern data
    :param file_appendix:   The appendix of the testpattern JSON data
    :return:                Path to the report folder containing hardware design and testpattern data
    """
    build_dir = find_project_root() / "build_test" / dut_name
    build_dir.mkdir(exist_ok=True, parents=True)

    file_name = f"{dut_name}{file_appendix}.json".lower()
    with open(build_dir / file_name, "w") as f0:
        json.dump(testpattern, f0, indent=1)

    return build_dir


def read_testpattern(dut_name: str, file_appendix: str) -> dict:
    """Reading the data as testpattern in the cocotb testbench
    :param dut_name:        The name of the Top Module DUT in the cocotb testbench (using dut._name)
    :param file_appendix:   (optional) appendix of the testpattern json file for specific testpattern
    :return:                Dictionary with testpattern for testing the DUT
    """
    path_to_file = find_project_root() / "build_test" / dut_name
    file_name = f"{dut_name}{file_appendix}.json".lower()
    with open(
        path_to_file / file_name,
        "r",
    ) as f:
        data = json.load(f)
    return data
