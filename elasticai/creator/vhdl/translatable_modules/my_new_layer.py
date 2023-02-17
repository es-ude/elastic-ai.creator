from nn.layers import MyLayer as _MyLayer
from vhdl.designs import MyHWDesign


class MyLayer(_MyLayer):
    """
    HWDesigns should take a dict of python built-in primitives (list[str], list[float], float, int, etc.),
    the HWDesign is responsible for creating the correct string representation for that data.
    """

    def _get_parameter(self) -> dict[str, str]:
        return {"weight": str(self.weight)}

    def translate(self) -> MyHWDesign:
        return MyHWDesign(**self._get_parameter())
