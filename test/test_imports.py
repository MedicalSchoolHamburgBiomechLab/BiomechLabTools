import unittest


class TestImports(unittest.TestCase):
    def test_import_pandas(self):
        try:
            import pandas
        except ImportError as e:
            self.fail(f"Failed to import pandas: {e}")

    def test_import_numpy(self):
        try:
            import numpy
        except ImportError as e:
            self.fail(f"Failed to import numpy: {e}")

    def test_import_matplotlib(self):
        try:
            import matplotlib
        except ImportError as e:
            self.fail(f"Failed to import matplotlib: {e}")

    def test_import_scipy(self):
        try:
            import scipy
        except ImportError as e:
            self.fail(f"Failed to import scipy: {e}")

    def test_import_dotenv(self):
        try:
            import dotenv
        except ImportError as e:
            self.fail(f"Failed to import dotenv: {e}")


if __name__ == '__main__':
    unittest.main()
