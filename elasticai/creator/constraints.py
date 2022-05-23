class WeightClipper:
    """
    This is an applied constraint meant to be used  after each batch. It clips the weights of the given module to be
     between -1,1.
    Meant to be used with binarization
    """

    def __init__(self):
        pass

    def __call__(self, module):
        module.weight.clip_(-1, 1)
