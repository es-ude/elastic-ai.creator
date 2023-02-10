import unittest
from pathlib import Path

import mypy.api

MYPY_ARGS = [
    str(Path("elasticai") / "creator" / "examples"),
]


class ExamplesTest(unittest.TestCase):
    def test_examples_using_mypy(self) -> None:
        reports, _, _ = mypy.api.run(MYPY_ARGS)
        if not reports.startswith("Success"):
            raise self.failureException(
                f"Typechecking failed. The following MyPy errors occured:\n{reports}"
            )
