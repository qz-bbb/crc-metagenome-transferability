import importlib.util
import sys
import unittest
from pathlib import Path
available=importlib.util.find_spec('prediction_extension.membership')
if available:from prediction_extension import membership as mod

class MembershipContracts(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(available,'Membership summary missing')
    def test_overlap(self):
        self.assertEqual(mod.overlap({'a','b'},{'b','c'}),(1,3,1/3))
        self.assertEqual(mod.overlap(set(),set()),(0,0,None))
    def test_undetected_and_zero_partition(self):
        self.assertEqual(mod.direction_category(False,True,False,False),'not_detected')
        self.assertEqual(mod.direction_category(True,True,False,False),'zero_difference')
        self.assertEqual(mod.direction_category(True,False,True,False),'concordant')
        self.assertEqual(mod.direction_category(True,False,False,True),'opposite')
        with self.assertRaises(ValueError):mod.direction_category(True,False,True,True)

if __name__=='__main__':unittest.main()
