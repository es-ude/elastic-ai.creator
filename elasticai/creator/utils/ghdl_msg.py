from dataclasses import dataclass
from typing import Literal


@dataclass
class GhdlMsg:
    type: Literal["assertion error", "simulation finished"]
    file: str
    line: int
    column: int
    time: str
    msg: str

    def render(self) -> str:
        return (
            "ghdl output:\n"
            f"\ttype: {self.type}\n"
            f"\tfile: {self.file}:{self.line}:{self.column}\n"
            f"\ttime: {self.time}\n"
            f"\tmsg: {self.msg}"
        )


def parse_ghdl_msg(txt: str) -> list[GhdlMsg]:
    msgs = []
    for line in txt.splitlines():
        parts = line.split(":", maxsplit=5)
        if len(parts) == 6:
            msg = GhdlMsg(
                type="assertion error",
                msg=parts[5].strip(),
                file=parts[0],
                line=int(parts[1]),
                column=int(parts[2]),
                time=parts[3][1:],
            )
            msgs.append(msg)
        elif line.startswith("simulation finished"):
            msgs.append(
                GhdlMsg(
                    type="simulation finished",
                    file=".",
                    line=-1,
                    column=-1,
                    time=line.strip("simulation finished @"),
                    msg="",
                )
            )
        elif len(msgs) > 1:
            msgs[-1].msg += line

    return msgs
