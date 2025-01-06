class Printer:
    def __init__(self):
        self._red = "31"
        self._green = "32"
        self._bold = "1"
        self._reset_all = "0"
        self._reset_bold = "22"

    @staticmethod
    def _cmd(*args: str) -> str:
        _args = ";".join(args)
        return f"\x1b[{_args}m"

    def red(self, txt: str) -> str:
        return self._colored(txt, self._red)

    def red_bold(self, txt: str):
        return self._colored_bold(txt, self._red)

    def green(self, txt: str) -> str:
        return self._colored(txt, self._green)

    def green_bold(self, txt: str) -> str:
        return self._colored_bold(txt, self._green)

    def _colored(self, txt: str, color: str):
        return "".join([self._cmd(color), txt, self._cmd(self._reset_all)])

    def _colored_bold(self, txt: str, color: str):
        return "".join([self._cmd(color, self._bold), txt, self._cmd(self._reset_all)])
