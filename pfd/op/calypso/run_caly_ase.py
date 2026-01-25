import json
import logging
import pickle
from pyexpat import model
import re
import shutil
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter,
    TransientError,
)

import numpy as np
from pfd.constants import (
    ase_input_name
)
from pfd.exploration import md
from pfd.exploration.md import (
    CalculatorWrapper,
    MDRunner
    )
from pfd.exploration.task import (
    ExplorationTaskGroup,
)
from pfd.utils import (
    BinaryFileInput,
    set_directory,
)
from pfd.utils.run_command import (
    run_command,
)

import glob
from ase.io import read,write

class RunCalyASEOptim(OP):
    r"""Perform structure optimization with DP in `ip["work_path"]`.

    The `optim_results_dir` and `traj_results` will be returned as `op["optim_results_dir"]`
    and `op["traj_results"]`.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": Parameter(str),  # calypso_task.idx
                "finished": Parameter(str),
                "cnt_num": Parameter(int),
                "task_dir": Artifact(Path),  # ready to run structure optimization
                "models": Artifact(List[Path]),  # model.ckpt.pt or frozen_model.pb
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "optim_results_dir": Artifact(Path),
                "traj_results": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `config`: (`dict`) The config of calypso task to obtain the command of calypso.
            - `task_name` : (`str`)
            - `finished` : (`str`)
            - `cnt_num` : (`int`)
            - `task_dir` : (`Path`)

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_name`: (`str`)
            - `optim_results_dir`: (`List[str]`)
            - `traj_results`: (`Artifact(List[Path])`)
        """
        finished = ip["finished"]
        cnt_num = ip["cnt_num"]

        task_path = ip["task_dir"]
        if task_path is not None:
            input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        else:
            input_files = []

        config = ip["config"] if ip["config"] is not None else {}
        #stress = config.pop("external_pressure", 0.0)  # GPa        

        # ==================== FIX BEGIN: 读取正确的压强 (120 GPa) ====================
        # 逻辑：CALYPSO Task Group 把压强写进了 ase_input.json 的 scalar_pressure 字段
        # 遍历 input_files 找到这个 json 文件并读取它
        stress = 0.0
        found_config = False
        
        # 尝试从输入文件中寻找 ase_input.json (即 ase_input_name)
        for fpath in input_files:
            if fpath.name == ase_input_name:
                try:
                    with open(fpath, 'r') as f:
                        ase_params = json.load(f)
                        # 读取 scalar_pressure
                        stress = ase_params.get("scalar_pressure", 0.0)
                        print(f"DEBUG: Successfully loaded pressure from {fpath.name}: {stress} GPa", flush=True)
                        found_config = True
                except Exception as e:
                    print(f"DEBUG: Error reading {fpath.name}: {e}", flush=True)
                break
        
        if not found_config:
            # Fallback: 如果没找到文件，尝试从 config 读，或者默认为 0
            # 这里同时兼容 external_pressure 和 pressure 写法
            stress = config.pop("pressure", config.pop("external_pressure", 0.0))
            print(f"DEBUG: ase_input.json not found, falling back to config pressure: {stress} GPa", flush=True)
        # 计算 Pullay stress (单位转换: GPa -> kbar)
        pstress = stress * 10 #GPa to kbar
        print(f"DEBUG: Config Pressure Read -> {stress} GPa", flush=True)
        # ==================== FIX END ====================

        models=ip["models"]
        model_files = [mm.resolve() for mm in models]

        work_dir = Path(ip["task_name"])
        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)

            if finished == "false":
                calc_style = config.pop("calculator", "mace")
                calc = CalculatorWrapper.get_calculator(calc_style)
                calc = calc().create(model_path=str(model_files[0]), **config)
                poscar_list = [poscar.resolve() for poscar in Path().glob("POSCAR_*")]                
                for poscar in poscar_list:
                    
                    atoms=read(poscar,index=0,format='vasp')
                    md_runner = MDRunner(atoms)
                    md_runner.set_calculator(calc)
                    try:
                        md_runner.run_md_from_json(
                            ase_input_name,
                            traj_file=poscar.name.strip("POSCAR_") + ".traj",
                            )
                    except Exception as e:
                        logging.error(f"ASE opt failed: {e}")
                        raise TransientError("ASE opt failed")

                # run opt                    
                    atoms_lat = atoms.cell
                    atoms_pos = atoms.positions
                    atoms_force = atoms.get_forces()
                    atoms_stress = atoms.get_stress()
                # eV/A^3 to GPa
                    atoms_stress = atoms_stress/(0.01*0.6242)
                    atoms_symbols = atoms.get_chemical_symbols()
                    atoms_ene = atoms.get_potential_energy()
                    atoms_vol = atoms.get_volume()
                    element, ele = self.Get_Element_Num(atoms_symbols)
                    outcar = poscar.name.replace("POSCAR", "OUTCAR")
                    contcar = poscar.name.replace("POSCAR", "CONTCAR")
                    
                    self.write_contar(contcar, element, ele, atoms_lat, atoms_pos)
                    self.write_outcar(outcar, element, ele, atoms_vol, atoms_lat, 
                                      atoms_pos, atoms_ene, atoms_force, 
                                      atoms_stress * -10.0, pstress)

                optim_results_dir = Path("optim_results_dir")
                optim_results_dir.mkdir(parents=True, exist_ok=True)
                for poscar in Path().glob("POSCAR_*"):
                    target = optim_results_dir.joinpath(poscar.name)
                    shutil.copyfile(poscar, target)
                for contcar in Path().glob("CONTCAR_*"):
                    target = optim_results_dir.joinpath(contcar.name)
                    shutil.copyfile(contcar, target)
                for outcar in Path().glob("OUTCAR_*"):
                    target = optim_results_dir.joinpath(outcar.name)
                    shutil.copyfile(outcar, target)

                traj_results_dir = Path("traj_results")
                traj_results_dir.mkdir(parents=True, exist_ok=True)
                for traj in Path().glob("*.traj"):
                    target = traj_results_dir.joinpath(str(cnt_num) + "." + traj.name)
                    shutil.copyfile(traj, target)

            else:
                optim_results_dir = Path("optim_results_dir")
                optim_results_dir.mkdir(parents=True, exist_ok=True)
                traj_results_dir = Path("traj_results")
                traj_results_dir.mkdir(parents=True, exist_ok=True)

        return OPIO(
            {
                "task_name": str(work_dir),
                "optim_results_dir": work_dir / optim_results_dir,
                "traj_results": work_dir / traj_results_dir,
            }
        )
    def write_contar(self, contcar, element, ele, lat, pos):
        '''Write CONTCAR
        '''
        with open(contcar, 'w', encoding='utf-8') as f:
            #f = open(contcar,'w',encoding='utf-8')
            f.write('ASE-DP-OPT\n')
            f.write('1.0\n')
            for i in range(3):
                f.write('%15.10f %15.10f %15.10f\n' % tuple(lat[i]))
            for x in element:
                f.write(x + '  ')
            f.write('\n')
            for x in element:
                f.write(str(ele[x]) + '  ')
            f.write('\n')
            f.write('Direct\n')
            na = sum(ele.values())
            dpos = np.dot(pos,np.linalg.inv(lat))
            for i in range(na):
                f.write('%15.10f %15.10f %15.10f\n' % tuple(dpos[i]))
            
    def Get_Element_Num(self, elements):
        '''Using the Atoms.get_chemical_symbols to Know Element&Num'''
        element = []
        ele = {}
        element.append(elements[0])
        for x in elements:
            if x not in element :
                element.append(x)
        for x in element:
            ele[x] = elements.count(x)
        return element, ele
    
    def write_outcar(self,outcar, element, ele, volume, lat, pos, ene, force, stress, pstress):
        '''Write OUTCAR'''
        #f = open(outcar,'w',encoding='utf-8')
        with open(outcar, 'w', encoding='utf-8') as f:
            for x in element:
                f.write('VRHFIN =' + str(x) + '\n')
            f.write('ions per type =')
            for x in element:
                f.write('%5d' % ele[x])
            f.write('\nDirection     XX             YY             ZZ             XY             YZ             ZX\n')
            f.write('in kB')
            f.write('%15.6f' % stress[0])
            f.write('%15.6f' % stress[1])
            f.write('%15.6f' % stress[2])
            f.write('%15.6f' % stress[3])
            f.write('%15.6f' % stress[4])
            f.write('%15.6f' % stress[5])
            f.write('\n')
            mean_stress = np.sum(stress[0] + stress[1] + stress[2])/3.0
            ext_pressure_diff = np.sum(stress[0] + stress[1] + stress[2])/3.0 - pstress
            f.write('external pressure = %20.6f kB    Pullay stress = %20.6f  kB\n'% (ext_pressure_diff, pstress))
            f.write('volume of cell : %20.6f\n' % volume)
            f.write('direct lattice vectors\n')
            for i in range(3):
                f.write('%10.6f %10.6f %10.6f\n' % tuple(lat[i]))
            f.write('POSITION                                       TOTAL-FORCE(eV/Angst)\n')
            f.write('-------------------------------------------------------------------\n')
            na = sum(ele.values())
            for i in range(na):
                f.write('%15.6f %15.6f %15.6f' % tuple(pos[i]))
                f.write('%15.6f %15.6f %15.6f\n' % tuple(force[i]))
            f.write('-------------------------------------------------------------------\n')
            f.write('energy  without entropy= %20.6f %20.6f\n' % (ene, ene/na))
            enthalpy = ene + pstress * volume / 1602.17733 # kb to ev/A^3
            f.write('enthalpy is  TOTEN    = %20.6f %20.6f\n' % (enthalpy, enthalpy/na))
            print("DEBUG: Testing my fix for enthalpy...", flush=True)
            print('enthalpy is  TOTEN    = %20.6f %20.6f\n' % (enthalpy, enthalpy/na), flush=True)
            print('pstress:', pstress, 'volume:', volume, 'ene:', ene, flush=True)
            print(f"DEBUG: Real Stress={mean_stress:.2f}, Target={pstress:.2f}, Written External={ext_pressure_diff:.2f}", flush=True)