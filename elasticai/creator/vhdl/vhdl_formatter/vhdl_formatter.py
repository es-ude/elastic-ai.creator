import subprocess

from elasticai.creator.resource_utils import get_full_path

"""
    we are using vsg (VHDL Style Guide) for indenting the generated vhdl files
    config.json defines the indentation size, since the default indentation size is 2 spaces

    source :
        https://github.com/jeremiah-c-leary/vhdl-style-guide/tree/master
"""


def format_vhdl(file_path: str):
    config_path = get_full_path("elasticai.creator.vhdl.vhdl_formatter", "config.json")
    subprocess.run(
        f"vsg -f {file_path} --style indent_only --configuration {config_path} --fix",
        shell=True,
    )
