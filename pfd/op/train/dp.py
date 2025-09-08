import random
from typing import Any, List, Dict, Union, Optional, Tuple
from pathlib import Path
import json
import sys
import os
import glob
import logging
import shutil
from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from ase.io import read, write
from dflow.python import (
    FatalError,
    TransientError
)
from .train import Train
from pfd.constants import (
    train_script_name
    )
from pfd.utils import (
    run_command,
    ase2multisys
)
class DPTrain(Train):
    """[Modified from DPGEN2 RunDPTrain]

    Args:
        Train (_type_): _description_
    """
    
    default_optional_parameter = {}
    train_script_name = "input.json"
    model_file = "model.pb"
    log_file = "train.log"
    lcurve_file = "lcurve.out"
    
    def _process_script(self, input_dict) -> Any:
        return self._script_rand_seed(input_dict)

    def run_train(
        self,
        config: Dict[str, Any],
        init_model: Optional[Union[str, Path]] = None,
        init_data: Optional[List[Path]] = None,
        iter_data: Optional[Path] = None,
        valid_data: Optional[Path] = None,
        train_dict: Dict[str, Any]={},
        optional_files: Optional[List[Path]]=None,
        #**kwargs
        ):
        train_dict = self._process_script(train_dict)
        if (not init_data or len(init_data) == 0) and not iter_data:
            raise FatalError(
                "At least one data source must be provided: "
                "either 'init_data' (non-empty list) or 'iter_data' (valid path)"
            )
        dp_command = config.get("command", "dp").split()
        impl = config.get("impl", "tensorflow")
        assert impl in ["tensorflow", "pytorch"]
        if impl == "pytorch":
            dp_command.append("--pt")
        finetune_args = config.get("finetune_args", "")
        train_args = config.get("train_args", "")
        config = DPTrain.normalize_config(config)
        
        finetune_mode = config.pop("finetune_mode",False)
        mixed_type = config.pop("mixed_type",False)
        # convert extxyz to dpdata systems...
        if init_data and len(init_data) > 0:
            ms=[]
            for ii in init_data:
                ms += read(ii,index=":")
            ms=ase2multisys(ms,labeled=True)
            if mixed_type:
                ms.to("deepmd/npy/mixed", "./init_data")
            else:
                ms.to("deepmd/npy", "./init_data")
        if iter_data:
            ms = read(iter_data, index=":")
            ms = ase2multisys(ms, labeled=True)
            if mixed_type:
                ms.to("deepmd/npy/mixed", "./iter_data")
            else:
                ms.to("deepmd/npy", "./iter_data")
        
        if valid_data:
            ms = read(valid_data, index=":")
            ms = ase2multisys(ms, labeled=True)
            if mixed_type:
                ms.to("deepmd/npy/mixed", "./valid_data")
            else:
                ms.to("deepmd/npy", "./valid_data")
                
            valid_data = _get_system_path("./valid_data")
        else:
            valid_data = None
            
        
        if "systems" in train_dict["training"]:
            major_version = "1"
        else:
            major_version = "2"
            
        # auto prob style
        do_init_model = False
        auto_prob_str = "prob_sys_size"
        
        train_dict = DPTrain.write_data_to_input_script(
            train_dict,
            config,
            _get_system_path("./init_data") if init_data and len(init_data) > 0 else [],
            _get_system_path("./iter_data") if iter_data else [],
            auto_prob_str,
            major_version,
            valid_data,
        )

        train_dict["training"]["disp_file"] = "lcurve.out"
        
        
        # open log
        fplog = open("train.log", "w")
        
        def clean_before_quit():
            fplog.close()
            
        with open(train_script_name, "w") as fp:
                json.dump(train_dict, fp, indent=4)

        if optional_files is not None:
            for f in optional_files:
                Path(f.name).symlink_to(f)
            
        command = DPTrain._make_train_command(
            dp_command,
            train_script_name,
            impl,
            do_init_model,
            init_model,
            finetune_mode,
            finetune_args,
            False,# init_model_with_finetune,
            train_args,
            )

        ret, out, err = run_command(command)
        if ret != 0:
            clean_before_quit()
            logging.error(
                    "".join(
                        (
                            "dp train failed\n",
                            "out msg: ",
                            out,
                            "\n",
                            "err msg: ",
                            err,
                            "\n",
                        )
                    )
                )
            raise FatalError("dp train failed")
        fplog.write("#=================== train std out ===================\n")
        fplog.write(out)
        fplog.write("#=================== train std err ===================\n")
        fplog.write(err)

        if finetune_mode == True and os.path.exists("input_v2_compat.json"):
            shutil.copy2("input_v2_compat.json", train_script_name)

        # freeze model
        if impl == "pytorch":
            self.model_file = "model.ckpt.pt"
        else:
            ret, out, err = run_command(["dp", "freeze", "-o", "frozen_model.pb"])
            if ret != 0:
                clean_before_quit()
                logging.error(
                        "".join(
                            (
                                "dp freeze failed\n",
                                "out msg: ",
                                out,
                                "\n",
                                "err msg: ",
                                err,
                                "\n",
                            )
                        )
                    )
                raise FatalError("dp freeze failed")
            self.model_file = "frozen_model.pb"
        fplog.write("#=================== freeze std out ===================\n")
        fplog.write(out)
        fplog.write("#=================== freeze std err ===================\n")
        fplog.write(err)
        clean_before_quit()

    def _set_desc_seed(self, desc):
        """Set descriptor seed.

        Args:
            desc (_type_): _description_
        """
        if desc["type"] == "hybrid":
            for desc in desc["list"]:
                self._set_desc_seed(desc)
        elif desc["type"] not in ["dpa1", "dpa2"]:
            desc["seed"] = random.randrange(sys.maxsize) % (2**32)

    def _script_rand_seed(
            self,
            input_dict,
        ):
        jtmp = input_dict.copy()
        if "model_dict" in jtmp["model"]:
            for d in jtmp["model"]["model_dict"].values():
                if isinstance(d["descriptor"], str):
                    self._set_desc_seed(jtmp["model"]["shared_dict"][d["descriptor"]])
                d["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        else:
            self._set_desc_seed(jtmp["model"]["descriptor"])
            jtmp["model"]["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (
                2**32
            )
        jtmp["training"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        return jtmp
    
    @staticmethod
    def write_data_to_input_script(
        idict: dict,
        config,
        init_data: Union[List[Path], Dict[str, List[Path]]],
        iter_data: List[Path],
        auto_prob_str: str = "prob_sys_size",
        major_version: str = "1",
        valid_data: Optional[List[Path]] = None,
    ):
        odict = idict.copy()
        data_list = [str(ii) for ii in init_data] + [str(ii) for ii in iter_data]
        if major_version == "1":
            # v1 behavior
            odict["training"]["systems"] = data_list
            odict["training"].setdefault("batch_size", "auto")
            odict["training"]["auto_prob_style"] = auto_prob_str
            if valid_data is not None:
                odict["training"]["validation_data"] = {
                    "systems": [str(ii) for ii in valid_data],
                    "batch_size": 1,
                }
        elif major_version == "2":
            # v2 behavior
            odict["training"]["training_data"]["systems"] = data_list
            odict["training"]["training_data"].setdefault("batch_size", "auto")
            odict["training"]["training_data"]["auto_prob"] = auto_prob_str
            if valid_data is None:
                odict["training"].pop("validation_data", None)
            else:
                odict["training"]["validation_data"] = {
                    "systems": [str(ii) for ii in valid_data],
                    "batch_size": 1,
                }
        else:
            raise RuntimeError("unsupported DeePMD-kit major version", major_version)
        return odict
    
    @staticmethod
    def _make_train_command(
        dp_command,
        train_script_name,
        impl,
        do_init_model,
        init_model,
        finetune_mode,
        finetune_args,
        init_model_with_finetune,
        train_args="",
        ):
        # find checkpoint
        if impl == "tensorflow" and os.path.isfile("checkpoint"):
            checkpoint = "model.ckpt"
        elif impl == "pytorch" and len(glob.glob("model.ckpt-[0-9]*.pt")) > 0:
            checkpoint = "model.ckpt-%s.pt" % max(
                [int(f[11:-3]) for f in glob.glob("model.ckpt-[0-9]*.pt")]
            )
        else:
            checkpoint = None
        # case of restart
        if checkpoint is not None:
            command = dp_command + ["train", "--restart", checkpoint, train_script_name]
            return command
        # case of init model and finetune
        assert checkpoint is None
        case_init_model = do_init_model and (not init_model_with_finetune)
        case_finetune = finetune_mode == True or (
            do_init_model and init_model_with_finetune
            )
        if case_init_model:
            init_flag = "--init-frz-model" if impl == "tensorflow" else "--init-model"
            command = dp_command + [
                "train",
                init_flag,
                str(init_model),
                train_script_name,
            ]
        elif case_finetune:
            command = (
            dp_command
            + [
                "train",
                train_script_name,
                "--finetune",
                str(init_model),
            ]
            + finetune_args.split()
            )
        else:
            command = dp_command + ["train", train_script_name]
        command += train_args.split()
        return command

    @staticmethod
    def training_args():
        doc_command = "The command for DP, 'dp' for default"
        doc_impl = "The implementation/backend of DP. It can be 'tensorflow' or 'pytorch'. 'tensorflow' for default."
        doc_finetune_args = "Extra arguments for finetuning"
        doc_multitask = "Do multitask training"
        doc_head = "Head to use in the multitask training"
        doc_train_args = "Extra arguments for dp train"
        doc_finetune_mode = "Whether to run in finetune mode"
        doc_mixed_type = "Whether to use mixed type system for training"
        return [
            Argument(
                "command",
                str,
                optional=True,
                default="dp",
                doc=doc_command,
            ),
            Argument(
                "impl",
                str,
                optional=True,
                default="tensorflow",
                doc=doc_impl,
                alias=["backend"],
            ),
            Argument(
                "finetune_args",
                str,
                optional=True,
                default="",
                doc=doc_finetune_args,
            ),
            Argument(
                "multitask",
                bool,
                optional=True,
                default=False,
                doc=doc_multitask,
            ),
            Argument(
                "head",
                str,
                optional=True,
                default=None,
                doc=doc_head,
            ),
            Argument(
                "train_args",
                str,
                optional=True,
                default="",
                doc=doc_train_args,
            ),
            Argument(
                "finetune_mode",
                bool,
                optional=True,
                default=False,
                doc=doc_finetune_mode,
            ),
            Argument(
                "mixed_type",
                bool,
                optional=True,
                default=False,
                doc=doc_mixed_type,
            ),
        ]
    @staticmethod
    def normalize_config(data={}):
        ta = DPTrain.training_args()

        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)

        return data

def _get_system_path(
    data_dir:Union[str,Path]
    ):
    return [Path(ii).parent for ii in glob.glob(str(data_dir) + "/**/type.raw",recursive=True)]




