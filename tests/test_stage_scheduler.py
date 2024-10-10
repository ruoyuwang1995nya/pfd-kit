import sys

sys.path.append("../")
import unittest
from pfd.op.stage import StageScheduler
import dpdata
from pathlib import Path
import numpy as np
from dflow.python import OPIO


class TestStageScheduler(unittest.TestCase):
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
        pert_sys = sys.perturb(
            pert_num=10,
            cell_pert_fraction=0.03,
            atom_pert_distance=0.1,
            atom_pert_style="normal",
        )
        pert_sys.to("deepmd/npy", "Si")
        self.systems = [Path("Si")]
        self.stages = [
            [
                {
                    "_comment": "group 1 stage 1 of finetune-exploration",
                    "conf_idx": [0],
                    "n_sample": 2,
                    "exploration": {
                        "type": "lmp-md",
                        "ensemble": "npt",
                        "dt": 0.005,
                        "nsteps": 100,
                        "temps": [500],
                        "press": [1],
                        "trj_freq": 20,
                    },
                    "max_sample": 10000,
                }
            ]
        ]

    def test_dp_lmp(self):
        op = StageScheduler()
        out = op.execute(
            OPIO(
                {
                    "stages": self.stages,
                    "idx_stage": 0,
                    "systems": self.systems,
                    "type_map": ["Si"],
                    "mass_map": [28.09],
                    "scheduler_config": {"model_style": "dp", "explore_style": "lmp"},
                }
            )
        )
        # [print(ii.files()) for ii in out["task_grp"].task_list]
        # sys = dpdata.System(out["pert_sys"][0], fmt="deepmd/npy")
        self.assertEqual(len(out["task_grp"]), 2)
        self.assertNotEqual(out["task_grp"][0].files(), out["task_grp"][1].files())


if __name__ == "__main__":
    unittest.main()
