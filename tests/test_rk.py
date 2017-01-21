# =========================================================================
#
# Imports
#
# =========================================================================
import unittest
from .context import rk

# =========================================================================
#
# Class definitions
#
# =========================================================================


class RKTestCase(unittest.TestCase):
    """Tests for `rk.py`."""

    # =========================================================================
    def test_adjust_for_output0(self):
        """Given dt=0.1, t=1, tf = 2, tout=1.5, does adjust_for_output do the
        right thing?
        """
        dt, output, done = rk.adjust_for_output(0.1, 1, 2, 1.5)
        self.assertListEqual([dt, output, done], [0.1, False, False], msg=None)

    # =========================================================================
    def test_adjust_for_output1(self):
        """Given dt=0.1, t=1, tf = 2, tout=1.01, does adjust_for_output do the
        right thing?
        """
        dt, output, done = rk.adjust_for_output(0.1, 1, 2, 1.01)
        self.assertAlmostEqual(dt, 0.01, places=7)
        self.assertListEqual([output, done], [True, False], msg=None)

    # =========================================================================
    def test_adjust_for_output2(self):
        """Given dt=0.1, t=1, tf = 1.001, tout=1.1, does adjust_for_output do
        the right thing?
        """
        dt, output, done = rk.adjust_for_output(0.1, 1, 1.001, 1.1)
        self.assertAlmostEqual(dt, 0.001, places=7)
        self.assertListEqual([output, done], [True, True], msg=None)


if __name__ == '__main__':
    unittest.main()
