import brevitas.nn as bnn
from torch import nn

import elasticai.creator.brevitas.brevitas_quantizers as bquant
from elasticai.creator.brevitas.brevitas_model_comparison import (
    BrevitasModelComparisonTestCase,
)
from elasticai.creator.brevitas.translation_functions.quantizers import (
    translate_binarize_layer,
    translate_ternarize_layer,
)
from elasticai.creator.layers import Binarize, Ternarize


class QuantizersTest(BrevitasModelComparisonTestCase):
    def test_binarize(self):
        layer = Binarize()
        target = nn.Sequential(bnn.QuantIdentity(act_quant=bquant.BinaryActivation))
        translated_layers = translate_binarize_layer(layer)
        translated_model = nn.Sequential(translated_layers)

        self.assertModelEqual(target, translated_model)

    def test_ternarize(self):
        layer = Ternarize()
        target = nn.Sequential(bnn.QuantIdentity(act_quant=bquant.TernaryActivation))
        translated_layers = translate_ternarize_layer(layer)
        translated_model = nn.Sequential(translated_layers)

        self.assertModelEqual(target, translated_model)
