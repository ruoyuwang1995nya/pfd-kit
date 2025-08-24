from pathlib import Path
import numpy as np
import copy
from typing import Union, Optional,List
import logging
from .eval_model import TestReport
from dargs import Argument
from ase import Atoms
from ase.io import read, write
from pfd.exploration.md import CalculatorWrapper
from pfd.exploration.inference.eval_model import EvalModel
from pfd.exploration.inference.util import get_mae, get_rmse

@EvalModel.register("deepmd")
@EvalModel.register("dp")
@EvalModel.register("mattersim")
@EvalModel.register("mace")
class EvalASE(EvalModel):
    def load_model(
        self, 
        model_type: str,
        model: Optional[Union[Path, str]]=None,
        **kwargs):
        calc = CalculatorWrapper.get_calculator(model_type)
        self._model = calc().create(model_path=str(model), **kwargs)

    def read_data(
        self, 
        data: List[Atoms],
        ):
        self._data = data

    def evaluate(
        self, name: str = "default", prefix: Union[Path, str] = "./", **kwargs
    ):
        if isinstance(prefix, str):
            prefix = Path(prefix)

        res = TestReport
        pred_e=[]
        lab_e = []
        pred_f=[]
        lab_f = []
        atom_num = []
        for atoms in self._data:
            # read labels
            lab_e.append(atoms.get_potential_energy())
            lab_f.append(atoms.get_forces().flatten())
            # make prediction
            atoms.calc=self._model
            pred_e.append(atoms.get_potential_energy())
            pred_f.append(atoms.get_forces().flatten())
            atom_num.append(atoms.get_number_of_atoms())

        atom_num = np.array(atom_num)

        # energy prediction
        pred_e = np.array(pred_e)
        lab_e = np.array(lab_e)

        pred_e_atom = pred_e / atom_num
        lab_e_atom = lab_e / atom_num

        # force prediction
        pred_f = np.hstack(pred_f)
        lab_f = np.hstack(lab_f)
        
        # atom number array
        
        np.savetxt(
            str(prefix / (name + ".energy.txt")),
            np.column_stack((lab_e, pred_e)),
            header='',
            comments='#',
            fmt="%.6f",
        )
        np.savetxt(
            str(prefix / (name + ".energy_per_atom.txt")),
            np.column_stack((lab_e_atom, pred_e_atom)),
            header='',
            comments='#',
            fmt="%.6f",
        )

        np.savetxt(
            str(prefix / (name + ".force.txt")),
            np.column_stack((lab_f, pred_f)),
            header='',
            comments='#',
            fmt="%.6f",
        )
        res = TestReport(
            name=name,
            system=self._data,
            numb_frame=len(self._data),
            atom_numb=atom_num,
            mae_e=get_mae(pred_e, lab_e),
            rmse_e=get_rmse(pred_e, lab_e),
            mae_e_atom=get_mae(pred_e_atom, lab_e_atom) ,
            rmse_e_atom=get_rmse(pred_e_atom, lab_e_atom),
            mae_f=get_mae(pred_f, lab_f),
            rmse_f=get_rmse(pred_f, lab_f),
            lab_e=lab_e,
            pred_e=pred_e,
            lab_f=lab_f,
            pred_f=pred_f,
        )

        logging.info("#### number of frames: %d" % res.numb_frame)
        logging.info("#### Energy MAE: %.06f" % res.mae_e)
        logging.info("#### Energy RMSE: %.06f" % res.rmse_e)
        logging.info("#### Energy MAE per atom: %.06f" % res.mae_e_atom)
        logging.info("#### Energy RMSE per atom: %.06f" % res.rmse_e_atom)
        logging.info("#### Force MAE: %.06f" % res.mae_f)
        logging.info("#### Force RMSE: %.06f" % res.rmse_f)
        # get virial prediction
        # Not implemented yet
        report = res.report()
        return res, report

    def inference(
        self, name: str = "structures.extxyz", prefix: Union[Path, str] = "./", **kwargs
    ):
        max_force = kwargs.pop("max_force", None)
        if isinstance(prefix, str):
            prefix = Path(prefix)
            
        for atoms in self._data:
            atoms.calc = self._model
            energy= atoms.get_potential_energy()
            atoms.info['energy'] = energy
            forces=atoms.get_forces()
            atoms.set_array('force', forces)
            # in voight order
            stress = atoms.get_stress()
            atoms.info['stress'] = stress
        n_frame = len(self._data)
        clean_ls = [i for i in range(n_frame)]
        logging.info("#### Number of frames: %d" % n_frame)
        if max_force:
            logging.info("#### Filtering by max force")
            for frame in range(n_frame):
                if abs(self._data[frame].arrays['force']).max() > max_force:
                    clean_ls.remove(frame)
            logging.info("After cleaning, %d frames left." % len(clean_ls))
        
        
        labeled_data_cl = [self._data[i] for i in clean_ls]
        if isinstance(prefix, str):
            prefix = Path(prefix)
        write(str(prefix / name), labeled_data_cl, format="extxyz")
        return prefix, prefix / name

    @classmethod
    def args(cls):
        doc_head = "The head of model in multi_task mode"
        return [Argument("head", str, optional=True, default=None, doc=doc_head)]

    @classmethod
    def doc(cls):
        return "Additional parameters for inference with Deep Potential models"
