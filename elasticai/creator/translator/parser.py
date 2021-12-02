from typing import Iterable, Any, Sequence


class callable_Block():
    def __init__(self, block):
        self.layers = block
        if hasattr(block[0], 'translation_callback'):
            self.translation_callback = block[0].translation_callback

    def submodel(self, x, layer=1):
        if layer == 0:
            return self.layers[layer](x)
        return self.layers[layer](self.submodel(x, layer=layer - 1))

    def __call__(self, x):
        return self.submodel(x, len(self.layers) - 1)


class Parser:
    def parse(self, model: Iterable[Any]) -> Sequence[Any]:
        things = [x for x in model]
        if len(things) == 0:
            return []
        else:
            return [1]
