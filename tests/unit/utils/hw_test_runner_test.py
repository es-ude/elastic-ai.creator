import re

from pytest import fixture

from elasticai.creator.utils.ghdl_msg import GhdlMsg, parse_ghdl_msg


def test_re_matches_filename():
    expr = re.compile(r".*(\w+_tb)\.vhd$")
    assert expr.match("my_folder/my_file_tb.vhd")


class GhdlMsgParsingBase:
    @fixture
    def parsed(self) -> list[GhdlMsg]:
        return parse_ghdl_msg(
            "elastic-ai.creator/elasticai/creator_plugins/padding/vhdl/padding_remover_tb.vhd:84:11:@380ps:(assertion error): expected single bit padded to 01000AAA but was 01000100\n"
            "some/other/path_tb.vhd:10:12:@24ps:(assertion error): expected a but was b"
        )

    def test_type(self, actual, expected):
        assert expected.type == actual.type

    def test_msg(self, actual, expected):
        assert expected.msg == actual.msg

    def test_file(self, actual, expected):
        assert expected.file == actual.file

    def test_line(self, actual, expected):
        assert expected.line == actual.line

    def test_column(self, actual, expected):
        assert expected.column == actual.column

    def test_time(self, actual, expected):
        assert expected.time == actual.time


class TestParseFirstLineGhdl(GhdlMsgParsingBase):
    @fixture
    def actual(self, parsed) -> str:
        return parsed[0]

    @fixture
    def expected(self) -> GhdlMsg:
        return GhdlMsg(
            type="assertion error",
            file="elastic-ai.creator/elasticai/creator_plugins/padding/vhdl/padding_remover_tb.vhd",
            line=84,
            column=11,
            time="380ps",
            msg="expected single bit padded to 01000AAA but was 01000100",
        )


class TestParseSecondLineGhdl(GhdlMsgParsingBase):
    @fixture
    def actual(self, parsed) -> str:
        return parsed[1]

    @fixture
    def expected(self) -> GhdlMsg:
        return GhdlMsg(
            type="assertion error",
            file="some/other/path_tb.vhd",
            line=10,
            column=12,
            time="24ps",
            msg="expected a but was b",
        )
