import unittest

from labtools.utils.cutting_tool import CuttingTool


class TestCuttingTool(unittest.TestCase):
    def test_equal_length_pieces(self):
        tool = CuttingTool(num_pieces=4)
        data = list(range(12))  # Example data array
        result = tool.cut(data)
        expected = [data[0:3], data[3:6], data[6:9], data[9:12]]
        self.assertEqual(result, expected)

    def test_fixed_length_pieces(self):
        tool = CuttingTool(num_pieces=3, length_pieces_ms=4)
        data = list(range(12))
        result = tool.cut(data)
        expected = [data[0:4], data[4:8], data[8:12]]
        self.assertEqual(result, expected)

    def test_varying_length_pieces(self):
        tool = CuttingTool(num_pieces=3, length_pieces_ms=[5, 3, 4])
        data = list(range(12))
        result = tool.cut(data)
        expected = [data[0:5], data[5:8], data[8:12]]
        self.assertEqual(result, expected)

    def test_invalid_input_types(self):
        with self.assertRaises(TypeError):
            CuttingTool(num_pieces='three')


# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()
