import re


def test_re_matches_filename():
    expr = re.compile(r".*(\w+_tb)\.vhd$")
    assert expr.match("my_folder/my_file_tb.vhd")
