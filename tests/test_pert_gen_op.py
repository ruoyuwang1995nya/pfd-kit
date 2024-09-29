import unittest
from pfd.op.pert_gen import PertGen
import dpdata
from pathlib import Path
import numpy as np
from dflow.python import OPIO


class TestPertGen(unittest.TestCase):
    def setUp(self):
        sys = dpdata.System(
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
        sys.to("vasp/poscar", "POSCAR", frame_idx=0)
        self.init_confs = [Path("POSCAR")]
        self.config = {
            "init_configurations": {"type": "file", "fmt": "vasp/poscar"},
            "pert_generation": [
                {
                    "conf_idx": [0],
                    "atom_pert_distance": 0.1,
                    "cell_pert_fraction": 0.03,
                    "pert_num": 5,
                }
            ],
        }

    def test_pert(self):
        op = PertGen()
        out = op.execute(OPIO({"init_confs": self.init_confs, "config": self.config}))
        sys = dpdata.System(out["pert_sys"][0], fmt="deepmd/npy")
        self.assertEqual(sys.get_nframes(), 5)


if __name__ == "__main__":
    unittest.main()
