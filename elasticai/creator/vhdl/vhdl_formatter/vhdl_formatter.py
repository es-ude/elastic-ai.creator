import os
import subprocess

from elasticai.creator.vhdl.generator.generator_functions import get_file_path_string

"""
    we are using vsg (VHDL Style Guide) for indenting the generated vhdl files
    config.json defines the indentation size, since the default indentation size is 2 spaces
    
    source : 
        https://github.com/jeremiah-c-leary/vhdl-style-guide/tree/master
"""


def format_vhdl(file_path):
    # get the path of config
    config = get_file_path_string(
        relative_path_from_project_root="elasticai/creator/vhdl/vhdl_formatter",
        file_name="config.json",
    )
    # check if the vhdl file exist !
    if os.path.isfile(file_path):
        subprocess.Popen(
            "vsg -f {file_path} --style indent_only --configuration {config} --fix".format(
                file_path=file_path, config=config
            ),
            shell=True,
        )
    else:
        print("no such a file")
