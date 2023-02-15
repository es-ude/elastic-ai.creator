import unittest
from pathlib import Path

import mypy.api

from elasticai.creator import examples

MYPY_ARGS = [
    str(Path(examples.__file__).parent),
]


class ExamplesTest(unittest.TestCase):
    def test_examples_using_mypy(self) -> None:
        reports, _, _ = mypy.api.run(MYPY_ARGS)
        if not reports.startswith("Success"):
            raise self.failureException(
                f"Typechecking failed. The following MyPy errors occured:\n{reports}"
            )
