from pfd.exploration.inference.eval_model import EvalModel
from pfd.exploration.inference.util import get_mae, get_rmse
from pathlib import Path
import dpdata
import numpy as np
import copy
from typing import Union


class DPTest(EvalModel):
    def load_model(self, model: Union[Path, str]):
        if isinstance(model, str):
            model = Path(model)
        self.model_path = model
        from deepmd.infer import DeepPot

        print("Loading model")
        self._model = DeepPot(model)
        print("Model loaded")

    def read_data(self, data, fmt="deepmd/npy", **kwargs):
        self._data = dpdata.LabeledSystem(data, fmt=fmt, **kwargs)

    def evaluate(
        self, name: str = "default", prefix: Union[Path, str] = "./", **kwargs
    ):
        if isinstance(prefix, str):
            prefix = Path(prefix)
        res = {}
        res["name"] = name
        new_labeled_data = self._data.predict(self.model_path, **kwargs)
        atom_num = self._data.get_natoms()
        res["atom_numb"] = atom_num
        res["numb_frame"] = self._data.get_nframes()
        res["details"] = {}
        # get energy prediction
        new_labeled_e = new_labeled_data.data["energies"].flatten()
        orig_labeled_e = self._data.data["energies"].flatten()
        res["details"]["train_e"] = (orig_labeled_e,)
        res["details"]["pred_e"] = (new_labeled_e,)
        np.savetxt(
            str(prefix / (name + ".energy.txt")),
            np.column_stack((orig_labeled_e, new_labeled_e)),
            fmt="%.6f",
        )
        np.savetxt(
            str(prefix / (name + ".energy_per_atom.txt")),
            np.column_stack((orig_labeled_e / atom_num, new_labeled_e / atom_num)),
            fmt="%.6f",
        )
        res["MAE_energy"] = get_mae(new_labeled_e, orig_labeled_e)
        res["RMSE_energy"] = get_rmse(new_labeled_e, orig_labeled_e)
        res["MAE_energy_per_at"] = get_mae(new_labeled_e, orig_labeled_e) / atom_num
        res["RMSE_energy_per_at"] = get_rmse(new_labeled_e, orig_labeled_e) / atom_num
        # get force error
        new_labeled_f = new_labeled_data.data["forces"].flatten()
        orig_labeled_f = self._data.data["forces"].flatten()
        res["details"]["train_f"] = (orig_labeled_f,)
        res["details"]["pred_f"] = (new_labeled_f,)
        np.savetxt(
            str(prefix / (name + ".force.txt")),
            np.column_stack((orig_labeled_f, new_labeled_f)),
            fmt="%.6f",
        )
        res["MAE_force"] = get_mae(new_labeled_f, orig_labeled_f)
        res["RMSE_force"] = get_rmse(new_labeled_f, orig_labeled_f)
        # get virial prediction
        if self._data.has_virial():
            new_labeled_v = new_labeled_data.data["virials"].flatten()
            orig_labeled_v = self._data.data["virials"].flatten()
            res["details"]["train_v"] = (orig_labeled_v,)
            res["details"]["pred_v"] = (new_labeled_v,)
            np.savetxt(
                str(prefix / (name + ".virial.txt")),
                np.column_stack((orig_labeled_f, new_labeled_f)),
                fmt="%.6f",
            )
            res["MAE_virial"] = get_mae(new_labeled_v, orig_labeled_v)
            res["RMSE_virial"] = get_rmse(new_labeled_v, orig_labeled_v)
        report = copy.copy(res)
        report.pop("details")
        return res, report
