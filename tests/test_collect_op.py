import unittest
import sys

sys.path.append("../")
from pfd.op.collect import CollectData
from dflow.python import OPIO
from fake_data_set import (
    fake_multi_sys,
    fake_system,
)
import os
from pathlib import Path
import dpdata


class TestCollectData(unittest.TestCase):
    def setUp(self):
        self.labeled_data = [Path("data0"), Path("data1")]
        # fake sys
        self.atom_name = "bar"
        self.natoms_0 = 3
        self.natoms_1 = 4
        self.nframes_0 = 6
        self.nframes_1 = 5
        ss_0 = fake_system(self.nframes_0, self.natoms_0, self.atom_name)
        ss_1 = fake_system(self.nframes_1, self.natoms_1, self.atom_name)
        # dump
        ss_0.to("deepmd/npy", "data0")
        ss_1.to("deepmd/npy", "data1")

    def test_basic(self):
        op = CollectData()
        out = op.execute(
            OPIO(
                {
                    "systems": self.labeled_data,
                    "type_map": None,
                    "optional_parameters": {},
                }
            )
        )
        ms = dpdata.MultiSystems()
        ss_0 = dpdata.System(out["systems"][0], "deepmd/npy")
        ss_1 = dpdata.System(out["systems"][1], "deepmd/npy")
        ms.append(ss_0)
        ms.append(ss_1)
        self.assertEqual(ms.get_nframes(), 11)

    def test_sample(self):
        op = CollectData()
        out = op.execute(
            OPIO(
                {
                    "systems": self.labeled_data,
                    "type_map": None,
                    "optional_parameters": {
                        "multi_sys_name": "test_sample",
                        "sample_conf": {"confs": [0], "n_sample": 3},
                    },
                }
            )
        )
        ms = dpdata.MultiSystems()
        for ii in out["multi_systems"]:
            for jj in ii.rglob("type.raw"):
                ms.append(dpdata.System(jj.parent, "deepmd/npy"))
        self.assertEqual(ms.get_nframes(), 3)


if __name__ == "__main__":
    unittest.main()
