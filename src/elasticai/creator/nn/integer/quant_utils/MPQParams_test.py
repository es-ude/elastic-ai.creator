import importlib
import sys
from types import ModuleType

import pytest

from elasticai.creator.nn.integer.quant_utils import QParams

MODULE = "elasticai.creator.nn.integer.quant_utils.MPQParams"


def import_fresh(monkeypatch) -> ModuleType:
    # Remove a cached import, so that the module body runs again.
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    return importlib.import_module(MODULE)


def test_import_warns_about_deprecation(monkeypatch) -> None:
    with pytest.warns(DeprecationWarning, match="MPQParams is deprecated"):
        import_fresh(monkeypatch)


def test_reexports_the_qparams_classes(monkeypatch) -> None:
    with pytest.warns(DeprecationWarning):
        module = import_fresh(monkeypatch)

    assert module.QParams is QParams
