import unittest
from math import sqrt
import numpy as np
from ase import Atoms
from pfd.exploration.selector import (
    DistanceConfFilter,
    BoxSkewnessConfFilter,
    BoxLengthFilter,
    ConfFilters,
)


def make_si_diamond(a: float = 5.44370237) -> Atoms:
    """Create a conventional 8-atom Si diamond cell matching legacy test geometry."""
    # Fractional positions of diamond conventional cell
    fracs = np.array([
        [0.25, 0.25, 0.0],
        [0.0, 0.5, 0.5],
        [0.75, 0.25, 0.75],
        [0.0, 0.0, 0.0],
        [0.25, 0.75, 0.75],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.5, 0.0, 0.5],
    ])
    cell = np.eye(3) * a
    positions = fracs @ cell
    return Atoms(symbols=["Si"] * 8, positions=positions, cell=cell, pbc=True)


class TestConfFilters(unittest.TestCase):
    def setUp(self):
        self.si = make_si_diamond()

    def test_distance_filter_basic(self):
        # With a modest custom safe distance the structure is safe
        f_ok = DistanceConfFilter(custom_safe_dist={"Si": 1.0})
        self.assertTrue(f_ok.check(self.si))

        # Large safe distance => threshold (dr) becomes larger than nearest neighbor distance => fail
        f_fail = DistanceConfFilter(custom_safe_dist={"Si": 5.0})
        self.assertFalse(f_fail.check(self.si))

    def test_distance_filter_ratio_override(self):
        # Fails at ratio 1.0
        f_fail = DistanceConfFilter(custom_safe_dist={"Si": 5.0}, safe_dist_ratio=1.0)
        self.assertFalse(f_fail.check(self.si))
        # Lower the ratio enough so distances become acceptable
        f_pass = DistanceConfFilter(custom_safe_dist={"Si": 5.0}, safe_dist_ratio=0.3)
        self.assertTrue(f_pass.check(self.si))

    def test_box_skewness_filter(self):
        # Highly skewed cell: second lattice vector has large x component
        skewed = Atoms(
            symbols=["Si"],
            positions=[[0, 0, 0]],
            cell=[
                [5.0, 0.0, 0.0],
                [9.0, 2.0, 0.0],  # 9 (x) vs 2 (y) => ratio 4.5 > tan(60)=1.732 -> fail
                [0.0, 0.0, 5.0],
            ],
            pbc=True,
        )
        f_skew = BoxSkewnessConfFilter(theta=60.0)
        self.assertFalse(f_skew.check(skewed))

        # Relax the criterion (very small theta triggers failure more easily). Use larger theta to pass.
        f_relaxed = BoxSkewnessConfFilter(theta=80.0)  # tan(80)=5.67 > 9/2 => pass
        self.assertTrue(f_relaxed.check(skewed))

    def test_box_length_filter(self):
        bad_len = Atoms(
            symbols=["Si"],
            positions=[[0, 0, 0]],
            cell=[
                [1.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],  # 10x difference vs a & c
                [0.0, 0.0, 1.0],
            ],
            pbc=True,
        )
        f_len = BoxLengthFilter(length_ratio=5.0)
        self.assertFalse(f_len.check(bad_len))

        f_len_relaxed = BoxLengthFilter(length_ratio=12.0)
        self.assertTrue(f_len_relaxed.check(bad_len))

    def test_combined_filters(self):
        # Create three structures: good, skewed, and length-bad
        good = self.si
        skewed = Atoms(
            symbols=["Si"],
            positions=[[0, 0, 0]],
            cell=[[5.0, 0.0, 0.0], [9.0, 2.0, 0.0], [0.0, 0.0, 5.0]],
            pbc=True,
        )
        bad_len = Atoms(
            symbols=["Si"],
            positions=[[0, 0, 0]],
            cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=True,
        )

        # Distance filter passes all (moderate threshold)
        f_dist = DistanceConfFilter(custom_safe_dist={"Si": 1.0})
        f_skew = BoxSkewnessConfFilter(theta=60.0)
        f_len = BoxLengthFilter(length_ratio=5.0)

        combo = ConfFilters().add(f_dist).add(f_skew).add(f_len)
        selected = combo.check([good, skewed, bad_len])
        # Only the good one survives
        self.assertEqual(len(selected), 1)
        self.assertTrue(np.allclose(selected[0].cell, good.cell))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
