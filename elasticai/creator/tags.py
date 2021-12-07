class Wrapper:
    def __init__(self, module):
        self.elasticai_tags = []
        self._module = module

    def __getattr__(self, item):
        if item == "elasticai_tags":
            return s
        return self._module.__getattr__(item)

    def __setattr__(self, key, value):
        self._module.__setattr__(key, value)



def tag_precomputed(module):
