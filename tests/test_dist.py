import unittest
import os
import json
import sys

sys.path.append("../")
import tempfile
from pfd.entrypoint.submit import FlowGen
from dflow.python import OPIO


class TestWorkDist(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # poscar
        self.poscar = os.path.join(self.test_dir, "foo.vasp")
        with open(self.poscar, "w") as f:
            f.write("foo")
        # model file
        self.model = os.path.join(self.test_dir, "foo.pt")
        with open(self.model, "w") as f:
            f.write("foo")
        # model train script file
        self.train_script = os.path.join(self.test_dir, "foo.json")
        with open(self.train_script, "w") as f:
            json.dump({"foo": "foo"}, f, indent=4)
        self.config_param = {
            "default_step_config": {
                "template_config": {
                    "image": "registry.dp.tech/dptech/deepmd-kit:2024Q1-d23cf3e"
                },
            },
            "task": {"type": "dist"},
            "inputs": {
                "type_map": ["foo"],
                "mass_map": [1.0],
                "teacher_models_paths": [self.model],
            },
            "conf_generation": {
                "init_configurations": {
                    "type": "file",
                    "prefix": "./",
                    "fmt": "vasp/poscar",
                    "files": [self.poscar],
                },
                "pert_generation": [
                    {
                        "conf_idx": [0],
                        "atom_pert_distance": 0.1,
                        "cell_pert_fraction": 0.03,
                        "pert_num": 5,
                    }
                ],
            },
            "exploration": {
                "init_training": True,
                "skip_aimd": True,
                "max_iter": 1,
                "converge_config": {"type": "energy_rmse", "RMSE": 0.01},
                "filter": [{"type": "distance"}],
                "type": "lmp",
                "config": {
                    "command": "lmp -var restart 0",
                    "shuffle_models": False,
                    "head": None,
                },
                "stages": [
                    [
                        {
                            "conf_idx": [0],
                            "n_sample": 1,
                            "exploration": {
                                "type": "lmp-md",
                                "ensemble": "npt",
                                "dt": 0.005,
                                "nsteps": 100,
                                "temps": [300],
                                "press": [1],
                                "trj_freq": 1,
                            },
                            "max_sample": 100,
                        }
                    ]
                ],
            },
            "train": {
                "comment": "Training script for downstream DeePMD model",
                "type": "dp",
                "config": {"impl": "pytorch", "init_model_policy": "no"},
                "template_script": self.train_script,
            },
        }

    os.environ["DFLOW_DEBUG"] = "1"

    def test_wf_io(self):
        assert FlowGen(self.config_param).submit(no_submission=True)


if __name__ == "__main__":
    unittest.main()
