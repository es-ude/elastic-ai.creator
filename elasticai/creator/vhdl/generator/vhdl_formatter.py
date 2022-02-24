import subprocess
import os


def format_vhdl(file_path: str) -> None:
    """
    we are using vsg (vhdl-style-guide) to indent the generated vhdl files
    source:
    https://github.com/jeremiah-c-leary/vhdl-style-guide
    """
    if os.path.isfile(file_path):
        subprocess.Popen(
            "vsg --style indent_only -f {file_path} --fix".format(file_path=file_path),
            shell=True,
        )
    else:
        print("no such a file")