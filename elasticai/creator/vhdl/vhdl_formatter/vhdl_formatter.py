import subprocess

from elasticai.creator.resource_utils import get_full_path
from elasticai.creator.vhdl import vhdl_formatter

"""
    we are using vsg (VHDL Style Guide) for indenting the generated vhdl files
    config.json defines the indentation size, since the default indentation size is 2 spaces

    source :
        https://github.com/jeremiah-c-leary/vhdl-style-guide/tree/master
"""


def format_vhdl(file_path):
    # get the path of config
    config_path = get_full_path(package=vhdl_formatter, file_name="config.json")
    # check if the vhdl file exist !
    subprocess.Popen(
        f"vsg -f {file_path} --style indent_only --configuration {config_path} --fix",
        shell=True,
    )
