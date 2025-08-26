from pathlib import (
    Path,
)
from typing import (
    Any,
    Dict,
    List,
    Set,
    Tuple,
    Union,
)

from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
import shutil
from dflow.python import (
    TransientError
)
from ase.io import write,read
from ase import Atoms

from pfd.constants import (
    fp_task_pattern,
    ase_conf_name,
    ase_log_name
)
from pfd.exploration import task
from pfd.exploration.md import CalculatorWrapper
from pfd.utils import (
    set_directory
)


from .prep_fp import PrepFp
from .run_fp import RunFp

class PrepFpASE(PrepFp):
    def _create_tasks(
        self, 
        confs: List[Atoms],
        config: Dict,
        #model_file: Path,
        ) -> Tuple[List[str], List[Path]]:
        batch_size = config.get("batch_size",100)
        task_names = []
        task_paths = []
        #batches = []
        for idx, i in enumerate(range(0, len(confs), batch_size)):
            batch = confs[i:i + batch_size]
            task_name = fp_task_pattern % idx
            task_path = Path(task_name)
            with set_directory(task_path):
                self.prep_task(batch)
            task_names.append(task_name)
            task_paths.append(task_path)
        return task_names,task_paths
    
    def prep_task(
        self,
        confs: List[Atoms],
        #model_file: Path,
        **kwargs
    ):
        r"""Define how one Vasp task is prepared.

        Parameters
        ----------
        conf_frame : List[ase.Atoms]
            One frame of configuration in the ase.Atoms format.
        """
        write(ase_conf_name,confs)
        #shutil.copy(model_file,"./")

class RunFpASE(RunFp):
    def input_files(self) -> List[str]:
        return [ase_conf_name]

    def optional_input_files(self) -> List[str]:
        return []
    
    def run_task(
        self,
        model_style: str,
        model_args: Dict,
        model: Path,
        ):
        confs = read(ase_conf_name,index=':')
        calc = CalculatorWrapper.get_calculator(model_style)
        calc = calc().create(model_path=str(model), **model_args)
        for atoms in confs:
            try:
                atoms.calc = calc
                atoms.get_potential_energy()
                atoms.get_forces()
            except Exception as e:
                raise TransientError(f"Calculator error: {e}")
        
        # write structures with labels
        out_name = "labeled_" + ase_conf_name
        write(out_name, confs)
        
        # write log
        with open(ase_log_name,'w') as f:
            f.write(f"Model: {model.name}\n")
            f.write(f"Number of structures: {len(confs)}\n")
        return out_name, ase_log_name
    
    @staticmethod
    def args():
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_model_style = "Distillation model style, e.g., 'mace', 'mattersim'."
        doc_model_args = "Model arguments, e.g., {'device':'cuda'},  etc."
        return [
            Argument("model_style", str, optional=True, default="dp", doc=doc_model_style,alias=['style','model']),
            Argument("model_args",
                Dict,
                optional=True,
                default={},
                doc=doc_model_args,
            )
        ]
class ASEInputs:
    def __init__(self, **kwargs):
        self.ase = "ase"
        
    @staticmethod
    def args():
        return []