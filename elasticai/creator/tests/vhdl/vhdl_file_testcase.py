import unittest


class GeneratedVHDLCodeTest(unittest.TestCase):
    @staticmethod
    def get_non_empty_lines(text):
        text = text.splitlines()
        return [x for x in text if len(x) > 0]

    def check_generated_code(self, expected_code, code, check_comments: bool = False):
        expected = self.get_non_empty_lines(expected_code)
        actual = self.get_non_empty_lines(code)
        if check_comments:
            expected = [line.strip() for line in expected if not line.isspace()]

            actual = [line.strip() for line in actual if not line.isspace()]
        else:
            expected = [
                line.strip()
                for line in expected
                if not line.startswith("--") and not line.isspace()
            ]

            actual = [
                line.strip()
                for line in actual
                if not line.startswith("--") and not line.isspace()
            ]
        self.assertEqual(expected, actual)

    expected_code = None
