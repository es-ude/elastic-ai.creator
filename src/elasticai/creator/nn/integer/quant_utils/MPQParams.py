"""Former mixed-precision variant of `QParams`, merged into `QParams.py`.

Kept so that existing imports keep working; import from
`elasticai.creator.nn.integer.quant_utils` instead.
"""

from elasticai.creator.nn.integer.quant_utils.QParams import (
    AsymmetricSignedQParams as AsymmetricSignedQParams,
)
from elasticai.creator.nn.integer.quant_utils.QParams import (
    AsymmetricUnsignedQParams as AsymmetricUnsignedQParams,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams as QParams
from elasticai.creator.nn.integer.quant_utils.QParams import (
    QuantScheme as QuantScheme,
)
from elasticai.creator.nn.integer.quant_utils.QParams import (
    SymmetricSignedQParams as SymmetricSignedQParams,
)
from elasticai.creator.nn.integer.quant_utils.QParams import (
    SymmetricUnsignedQParams as SymmetricUnsignedQParams,
)
