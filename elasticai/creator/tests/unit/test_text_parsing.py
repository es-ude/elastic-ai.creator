import unittest

from elasticai.creator.tests.code_utilities_for_testing import CodeTestCase


class ExtractSectionTest(unittest.TestCase):
    def test_extract_CD_from_AACBAADB(self):
        text = [f"{c}" for c in "AACBAADB"]
        self.assertEqual(
            [["C"], ["D"]],
            CodeTestCase.extract_section_from_code(
                begin=["A", "A"], end="B", lines=text
            ),
        )

    def test_extract_CD_from_AACBBAADBB(self):
        text = [f"{c}" for c in "AACBBAADBB"]
        self.assertEqual(
            [["C"], ["D"]],
            CodeTestCase.extract_section_from_code(
                begin=["A", "A"], end=["B", "B"], lines=text
            ),
        )

    def test_extract_CD_from_ABCAABDAA(self):
        text = [f"{c}" for c in "ABCAABDAA"]
        self.assertEqual(
            [["C"], ["D"]],
            CodeTestCase.extract_section_from_code(
                begin=["A", "B"], end=["A", "A"], lines=text
            ),
        )
