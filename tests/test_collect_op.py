import unittest
from pdf.op.collect import CollectData
from dflow.python import OPIO
from fake_data_set import (
    fake_multi_sys,
    fake_system,
)
from pathlib import Path
import dpdata

from tests import fake_data_set


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

    def test_collect(self):
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
        ss_0 = dpdata.System(out["systems"][0], "deepmd/npy")
        self.assertEqual(ss_0.get_nframes(), 5)
        ss_1 = dpdata.System(out["systems"][1], "deepmd/npy")
        self.assertEqual(ss_1.get_nframes(), 6)


if __name__ == "__main__":
    print(fake_data_set.__path__)
    unittest.main()
