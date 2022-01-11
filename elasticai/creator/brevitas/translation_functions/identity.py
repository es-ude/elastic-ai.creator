from torch.nn import Module


def translate_layer(layer: Module) -> Module:
    """
    translation of a layer which stays the same, the same layer is just returned
    Args:
        layer (Layer): PyTorch layer
    Returns:
        input layer
    """
    return layer
