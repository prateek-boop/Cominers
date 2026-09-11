import unittest
from datetime import datetime, timezone
from scripts.prepare_cic2017 import capture_time, valid_ip

class CIC2017ConversionTests(unittest.TestCase):
    def test_day_first_and_missing_afternoon_marker(self):
        expected=datetime(2017,7,6,13,0,tzinfo=timezone.utc).timestamp()
        self.assertEqual(capture_time('6/7/2017 1:00',6),expected)
        self.assertEqual(capture_time('06/07/2017 13:00:00',6),expected)
        self.assertEqual(capture_time('03/07/2017 08:55:58',3),datetime(2017,7,3,8,55,58,tzinfo=timezone.utc).timestamp())
    def test_wrong_date_and_invalid_ip_rejected(self):
        with self.assertRaises(ValueError): capture_time('7/6/2017 1:00',6)
        with self.assertRaises(ValueError): valid_ip('not-an-ip')
