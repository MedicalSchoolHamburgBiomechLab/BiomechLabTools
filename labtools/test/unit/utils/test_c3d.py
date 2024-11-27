import unittest
from pathlib import Path

from labtools.utils.c3d import load_c3d


class TestC3D(unittest.TestCase):
    def test_load_c3d(self):
        file_path = Path("../../data/c3d_testfile.c3d")
        self.assertTrue(file_path.exists())

        data, meta = load_c3d(file_path)
        self.assertEqual(data['analog_rate'], 0.0)
        self.assertEqual(data['point_rate'], 300)
        self.assertEqual(data['marker']['SIPS_left'].shape, (240, 3))
        self.assertEqual(data['marker']['SIPS_right'].shape, (240, 3))
        self.assertEqual(data['analog'].shape, (240, 0))
        self.assertEqual(data['creation_date'], '2023-05-04')
        self.assertEqual(data['creation_time'], '10:53:24')
