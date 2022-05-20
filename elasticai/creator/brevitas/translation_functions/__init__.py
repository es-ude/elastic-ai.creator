"""
make conversion function of each layer usable in other packages
"""
from elasticai.creator.brevitas.translation_functions.conv import (
    translate_conv1d,
    translate_conv2d,
)
from elasticai.creator.brevitas.translation_functions.identity import translate_layer
from elasticai.creator.brevitas.translation_functions.linear import (
    translate_linear_layer,
)
from elasticai.creator.brevitas.translation_functions.quantizers import (
    translate_binarize_layer,
    translate_ternarize_layer,
)
