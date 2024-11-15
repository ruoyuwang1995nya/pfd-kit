from pfd.exploration.inference.eval_model import EvalModel
from pfd.exploration.inference.util import get_mae, get_rmse
from pathlib import Path
import dpdata
import numpy as np
import copy
from typing import Union, Optional
import logging
import os
from .eval_model import TestReport
from dargs import Argument


@EvalModel.register("dp")
@EvalModel.register("deepmd")
class DPTest(EvalModel):
    def load_model(self, model: Union[Path, str], **kwargs):
        if isinstance(model, str):
            model = Path(model)
        self.model_path = model
        from deepmd.infer import DeepPot

        print("Loading model")
        self._model = DeepPot(model, **kwargs)
        print("Model loaded")

    def read_data(self, data, fmt="deepmd/npy", **kwargs):
        self._data = dpdata.LabeledSystem(data, fmt=fmt, **kwargs)

    def read_data_unlabeled(self, data, fmt="deepmd/npy", **kwargs):
        self._data = dpdata.System(data, fmt=fmt, **kwargs)

    def evaluate(
        self, name: str = "default", prefix: Union[Path, str] = "./", **kwargs
    ):
        if isinstance(prefix, str):
            prefix = Path(prefix)

        res = TestReport
        new_labeled_data = self._data.predict(self.model, **kwargs)
        atom_num = self._data.get_natoms()
        # get energy prediction
        new_labeled_e = new_labeled_data.data["energies"].flatten()
        orig_labeled_e = self._data.data["energies"].flatten()
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

        # get force error
        new_labeled_f = new_labeled_data.data["forces"].flatten()
        orig_labeled_f = self._data.data["forces"].flatten()
        np.savetxt(
            str(prefix / (name + ".force.txt")),
            np.column_stack((orig_labeled_f, new_labeled_f)),
            fmt="%.6f",
        )
        res = TestReport(
            name=name,
            system=self._data,
            numb_frame=self._data.get_nframes(),
            atom_numb=atom_num,
            mae_e=get_mae(new_labeled_e, orig_labeled_e),
            rmse_e=get_rmse(new_labeled_e, orig_labeled_e),
            mae_e_atom=get_mae(new_labeled_e, orig_labeled_e) / atom_num,
            rmse_e_atom=get_rmse(new_labeled_e, orig_labeled_e) / atom_num,
            mae_f=get_mae(new_labeled_f, orig_labeled_f),
            rmse_f=get_rmse(new_labeled_f, orig_labeled_f),
            lab_e=orig_labeled_e,
            pred_e=new_labeled_e,
            lab_f=orig_labeled_f,
            pred_f=new_labeled_f,
        )

        logging.info("#### atom numbers: %d" % res.atom_numb)
        logging.info("#### number of frames: %d" % res.numb_frame)
        logging.info("#### Energy MAE: %.06f" % res.mae_e)
        logging.info("#### Energy RMSE: %.06f" % res.rmse_e)
        logging.info("#### Energy MAE per atom: %.06f" % res.mae_e_atom)
        logging.info("#### Energy RMSE per atom: %.06f" % res.rmse_e_atom)
        logging.info("#### Force MAE: %.06f" % res.mae_f)
        logging.info("#### Force RMSE: %.06f" % res.rmse_f)
        # get virial prediction
        if self._data.has_virial():
            new_labeled_v = new_labeled_data.data["virials"].flatten()
            orig_labeled_v = self._data.data["virials"].flatten()
            res.lab_v = orig_labeled_v
            res.pred_v = new_labeled_v
            np.savetxt(
                str(prefix / (name + ".virial.txt")),
                np.column_stack((orig_labeled_f, new_labeled_f)),
                fmt="%.6f",
            )
            logging.info("#### Virial MAE: %.06f" % res.mae_v)
            logging.info("#### Virial RMSE: %.06f" % res.rmse_v)
        report = res.report()
        return res, report

    def inference(
        self, name: str = "default", prefix: Union[Path, str] = "./", **kwargs
    ):
        max_force = kwargs.pop("max_force", None)
        if isinstance(prefix, str):
            prefix = Path(prefix)
        labeled_data = self._data.predict(self.model, **kwargs)
        cells = labeled_data.data["cells"]
        coords = labeled_data.data["coords"]
        energies = labeled_data.data["energies"]
        forces = labeled_data.data["forces"]
        virials = labeled_data.data["virials"]
        labeled_data_dict = copy.deepcopy(labeled_data.data)
        n_atom = sum(labeled_data_dict["atom_numbs"])
        n_frame = cells.shape[0]
        clean_ls = [i for i in range(n_frame)]
        logging.info("#### Number of frames: %d" % n_frame)
        if max_force:
            logging.info("#### Filtering by max force")
            for frame in range(n_frame):
                if abs(forces[frame]).max() > max_force:
                    clean_ls.remove(frame)
            logging.info("After cleaning, %d frames left." % len(clean_ls))
        logging.info(
            "#### Max energy per atom (eV/atom): %.06f"
            % (energies[clean_ls].max() / n_atom)
        )
        logging.info(
            "#### Min energy per atom (eV/atom): %.06f"
            % (energies[clean_ls].min() / n_atom)
        )
        labeled_data_dict["cells"] = cells[clean_ls]
        labeled_data_dict["coords"] = coords[clean_ls]
        labeled_data_dict["energies"] = energies[clean_ls]
        labeled_data_dict["forces"] = forces[clean_ls]
        labeled_data_dict["virials"] = virials[clean_ls]
        labeled_data_cl = dpdata.LabeledSystem(data=labeled_data_dict)
        if isinstance(prefix, str):
            prefix = Path(prefix)
        labeled_data_cl.to("deepmd/npy", str(prefix / name))

        return prefix, prefix / name

    @classmethod
    def args(cls):
        doc_head = "The head of model in multi_task mode"
        return [Argument("head", str, optional=True, default=None, doc=doc_head)]

    @classmethod
    def doc(cls):
        return "Additional parameters for inference with Deep Potential models"
