import unittest
import dpdata
from pfd.exploration.selector import DistanceConfFilter
import numpy as np
from dflow.python import OPIO


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.sys = dpdata.System(
            data={
                "atom_numbs": [8],
                "atom_names": ["Si"],
                "atom_types": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                "orig": np.array([0.0, 0.0, 0.0]),
                "cells": np.array(
                    [
                        [
                            [5.44370237e00, 2.36649176e-48, 0.00000000e00],
                            [9.00000000e-16, 5.44370237e00, 0.00000000e00],
                            [3.00000000e-16, 3.00000000e-16, 5.44370237e00],
                        ]
                    ]
                ),
                "coords": np.array(
                    [
                        [
                            [4.08277678e00, 4.08277678e00, 1.36092559e00],
                            [6.00000000e-16, 2.72185119e00, 2.72185119e00],
                            [4.08277678e00, 1.36092559e00, 4.08277678e00],
                            [0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [1.36092559e00, 4.08277678e00, 4.08277678e00],
                            [2.72185119e00, 2.72185119e00, 0.00000000e00],
                            [1.36092559e00, 1.36092559e00, 1.36092559e00],
                            [2.72185119e00, 1.50000000e-16, 2.72185119e00],
                        ]
                    ]
                ),
            }
        )

    def test_filter_dist(self):
        config = {"custom_safe_dist": {"Si": 1.0}}
        filter = DistanceConfFilter(**config)
        self.assertTrue(filter.check(self.sys))

        config = {"custom_safe_dist": {"Si": 5.0}}
        filter = DistanceConfFilter(**config)
        self.assertFalse(filter.check(self.sys))


if __name__ == "__main__":
    unittest.main()
