from lark import Transformer

from elasticai.creator.hdl.vhdl.language import Connection, Connections, Port, SignalDef

from .standalone_parser import Lark_StandAlone


class TreeToVHDL(Transformer):
    def connection(self, s):
        return Connection(_to=s[0].value, _from=s[1].value)

    def connections(self, s):
        return Connections(s)

    def port(self, s):
        return Port(s[0])

    def vhdl(self, s):
        return s[0]

    def signal_defs(self, s):
        return s

    def signal_type(self, s):
        v = s[0].data
        if v.value == "std_logic":
            return 0
        else:
            return int(s[1].value) - 1

    def signal_def(self, s):
        name, direction, width = s
        name = name.value
        direction = direction.data
        return SignalDef(name=name, direction=direction, width=width)


parser = Lark_StandAlone(transformer=TreeToVHDL())
