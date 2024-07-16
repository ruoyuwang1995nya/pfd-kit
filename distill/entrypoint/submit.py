from copy import deepcopy
import os
from typing import (
    Dict,
    List,
    Optional,
    Tuple
)
import json
import dpdata
import dflow
import dpgen2
import distill

from dflow import (
    ArgoStep,
    Step,
    Steps,
    Workflow,
    upload_artifact
)



from distill.entrypoint.args import normalize as normalize_args
from distill.entrypoint.common import (
    global_config_workflow,
)

from dpgen2.fp import(
    fp_styles
    
)

from dpgen2.superop import (
    PrepRunLmp,
    PrepRunDPTrain,
    PrepRunFp
    )

from dpgen2.op import (
    PrepLmp,
    PrepDPTrain,
    RunDPTrain
)

from distill.flow.fine_tune import FineTune
from distill.op import (
    PertGen,
    TaskGen,
    CollectData,
    Inference,
    RunLmp
)

from distill.superop import ExplorationBlock
from distill.flow import Distillation

from distill.constants import default_image
from dpgen2.utils.step_config import normalize as normalize_step_dict
from dpgen2.utils import (
    upload_artifact_and_print_uri
)

default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        }
    }
)



def make_dist_op(
    prep_lmp_step_config:dict,
    run_lmp_step_config:dict,
    prep_train_step_config:dict,
    run_train_step_config:dict,
    gen_lmp_step_config:dict,
    collect_data_step_config:dict,
    pert_gen_step_config: dict,
    inference_step_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None
    
):
    """
    Make a super OP template for distillation process
    """
    # build lmp op
    prep_run_lmp_op=PrepRunLmp(
        "prep-run-lmp-step",
        PrepLmp,
        RunLmp,
        prep_config=prep_lmp_step_config,
        run_config=run_lmp_step_config,
        upload_python_packages=upload_python_packages
    )
    
    expl_block_op= ExplorationBlock(
        "exploration",
        TaskGen,
        prep_run_lmp_op,
        CollectData,
        gen_lmp_step_config,
        collect_data_step_config,
        upload_python_packages=upload_python_packages
    )
    
    prep_run_dp_op = PrepRunDPTrain(
        "prep-run-dp-step",
        PrepDPTrain,
        RunDPTrain,
        prep_config=prep_train_step_config,
        run_config=run_train_step_config,
        upload_python_packages=upload_python_packages
    ) 
    
    dist_op = Distillation(
        "distillation",
        PertGen,
        expl_block_op,
        Inference,
        prep_run_dp_op,
        pert_gen_step_config,
        inference_step_config,
        upload_python_packages=upload_python_packages
    )
    return dist_op

def make_ft_op(
    fp_style: str = "vasp",
    pert_gen_step_config: dict = default_config,
    prep_fp_config: dict = default_config,
    run_fp_config: dict = default_config,
    prep_train_config:dict = default_config,
    run_train_config: dict = default_config,
    collect_data_step_config:dict= default_config,
    inference_step_config: dict= default_config,
    upload_python_packages: Optional[List[os.PathLike]] = None
):
    if fp_style in fp_styles.keys():
        prep_run_fp_op = PrepRunFp(
            "prep-run-fp",
            fp_styles[fp_style]["prep"],
            fp_styles[fp_style]["run"],
            prep_config=prep_fp_config,
            run_config=run_fp_config,
            upload_python_packages=upload_python_packages,
        )
    else:
        raise RuntimeError(f"unknown fp_style {fp_style}")

    finetune_op = PrepRunDPTrain(
        "finetune",
        PrepDPTrain,
        RunDPTrain,
        prep_config=prep_train_config,
        run_config=run_train_config,
        upload_python_packages=upload_python_packages,
        #finetune=False,
        valid_data=None,
    )
    print(inference_step_config)
    ft_op= FineTune(
        name="fine-tune",
        pert_gen_op=PertGen,
        prep_run_fp_op=prep_run_fp_op,
        collect_data_op=CollectData,
        prep_run_dp_op=finetune_op,
        inference_op=Inference,
        pert_gen_step_config=pert_gen_step_config,
        collect_data_step_config=collect_data_step_config,
        inference_step_config=inference_step_config,
        upload_python_packages = upload_python_packages
        
    )
    return ft_op



def get_systems_from_data(data, data_prefix=None):
    data = [data] if isinstance(data, str) else data
    assert isinstance(data, list)
    if data_prefix is not None:
        data = [os.path.join(data_prefix, ii) for ii in data]
    return data


def workflow_dist(
    config_dict:Dict,
)-> Step:
    """
    Build a workflow from the OP templates of distillation
    """
    ## get input config 
    default_step_config=config_dict["default_step_config"]
    prep_train_step_config=config_dict["step_configs"].get("prep_train_config",default_step_config)
    run_train_step_config=config_dict["step_configs"].get("run_train_config",default_step_config)
    prep_lmp_step_config = config_dict["step_configs"].get("prep_explore_config",default_step_config)
    run_lmp_step_config = config_dict["step_configs"].get("run_explore_config",default_step_config)
    gen_lmp_step_config = config_dict["step_configs"].get("gen_lmp_config",default_step_config)
    collect_data_step_config = config_dict["step_configs"].get("collect_data_config",default_step_config)
    pert_gen_step_config = config_dict["step_configs"].get("pert_gen_config",default_step_config)
    inference_step_config= config_dict["step_configs"].get("inference_config",default_step_config)
    
    # uploaded python packages
    upload_python_packages=[]
    upload_python_packages.extend(list(dpdata.__path__))
    upload_python_packages.extend(list(dflow.__path__))
    upload_python_packages.extend(list(distill.__path__))
    upload_python_packages.extend(list(dpgen2.__path__))
    
    ## task configs
    config_dict_total=deepcopy(config_dict)
    type_map=config_dict["inputs"]["type_map"]
    train_config=config_dict["train"]["config"]
    explore_config=config_dict["conf_generation"]["config"]
    inference_config=config_dict["inference"]
    dp_test_config=deepcopy(inference_config)
    dp_test_config["task"]="dp_test"
    
    
    ## prepare artifacts
    # read training template
    with open(config_dict["train"]["template_script"],'r') as fp:
        template_script=json.load(fp)
    init_confs=upload_artifact(config_dict["conf_generation"]["init_configurations"]["files"])
    teacher_model = upload_artifact([config_dict["inputs"]["teacher_model"]])
    iter_data = upload_artifact([])

    
    # make distillation op
    dist_op=make_dist_op(
        prep_lmp_step_config,
        run_lmp_step_config,
        prep_train_step_config,
        run_train_step_config,
        gen_lmp_step_config,
        collect_data_step_config,
        pert_gen_step_config,
        inference_step_config,
        upload_python_packages = upload_python_packages
    )
    
    # make distillation steps
    dist_step = Step(
        "dp-dist-step",
        template=dist_op,
        parameters={
            "block_id": "test-flow",
            "type_map": type_map,
            "config":config_dict_total, # Total input parameter file: to be changed in the future
            "numb_models": 1,#InputParameter(type=int),
            "template_script": template_script,
            "train_config": train_config,
            "explore_config": explore_config,
            "inference_config": inference_config,
            "inference_validation_config": inference_config,
            "dp_test_validation_config": dp_test_config
            },
        artifacts={
            "init_confs": init_confs,
            "teacher_model" : teacher_model,
            "iter_data": iter_data,
            #"validation_data": InputArtifact(optional=True)
        }
        
    )
    return dist_step

def workflow_finetune(
    config:Dict
)-> Step:
    """
    Build a workflow from the OP templates of distillation
    """
    ## get input config 
    fp_style = config["fp"]["type"]
    default_config=config["default_step_config"]
    prep_train_config=config["step_configs"].get("prep_train_config",default_config)
    run_train_config=config["step_configs"].get("run_train_config",default_config)
    prep_fp_config=config["step_configs"].get("perp_fp_config",default_config)
    run_fp_config=config["step_configs"].get("run_fp_config",default_config)
    run_collect_data_config = config["step_configs"].get("collect_data_config",default_config)
    run_pert_gen_config = config["step_configs"].get("pert_gen_config",default_config)
    run_inference_config= config["step_configs"].get("inference_config",default_config)
    
    # uploaded python packages
    upload_python_packages=[]
    upload_python_packages.extend(list(dpdata.__path__))
    upload_python_packages.extend(list(dflow.__path__))
    upload_python_packages.extend(list(distill.__path__))
    upload_python_packages.extend(list(dpgen2.__path__))
    
    ## task configs
    config_dict_total=deepcopy(config)
    type_map=config["inputs"]["type_map"]
    train_config=config["train"]["config"]
    inference_config=config["inference"]
    #dp_test_config=deepcopy(inference_config)
    #dp_test_config["task"]="dp_test"
    collect_data_config={}
    collect_data_config["test_size"]=config["conf_generation"].get("test_data",0.05)
    collect_data_config["system_partition"]=config["conf_generation"].get("system_partition",False)
    collect_data_config["labeled_data"]=True
    
    
    ## prepare artifacts
    # read training template
    with open(config["train"]["template_script"],'r') as fp:
        template_script=json.load(fp)
    init_confs=upload_artifact(config["conf_generation"]["init_configurations"]["files"])
    iter_data = upload_artifact([])
    init_models_paths = config["train"].get("init_models_paths", None)
    
    if init_models_paths is not None:
        init_models = upload_artifact_and_print_uri(init_models_paths, "init_models")
    else:
        raise FileNotFoundError("Pre-trained model must exist!")
    
    fp_config = {}
    fp_inputs_config = config["fp"]["inputs_config"]
    fp_inputs = fp_styles[fp_style]["inputs"](**fp_inputs_config)
    fp_config["inputs"] = fp_inputs
    fp_config["run"] = config["fp"]["run_config"]
    
    # make distillation op
    ft_op=make_ft_op(
        fp_style="vasp",
        pert_gen_step_config=run_pert_gen_config,
        prep_fp_config=prep_fp_config,
        run_fp_config=run_fp_config,
        prep_train_config=prep_train_config,
        run_train_config=run_train_config,
        collect_data_step_config=run_collect_data_config,
        inference_step_config=run_inference_config,
        upload_python_packages = upload_python_packages
    )
    
    ft_step= Step(
        "dp-dist-step",
        template=ft_op,
        parameters={
            "block_id": "test-flow",
            "type_map": type_map,
            "config":config_dict_total, # Total input parameter file: to be changed in the future
            "numb_models": 1,#InputParameter(type=int),
            "template_script": template_script,
            "train_config": train_config,
            "inference_config": inference_config,#{"task":"dp_test"},
            "fp_config":fp_config,
            "collect_data_config":collect_data_config
            },
        artifacts={
            "init_models": init_models,
            "init_confs": init_confs,
            "iter_data": iter_data,
        }
        
        
    )
    return ft_step

def submit_dist(
    wf_config,
    reuse_step: Optional[List[ArgoStep]] = None,
    replace_scheduler: bool = False,
    no_submission: bool = False,
):  
    """
    Major entry point for the whole workflow, only one config dict
    """
    # normalize args
    wf_config = normalize_args(wf_config)
    print(wf_config)

    global_config_workflow(wf_config)
    
    dist_step=workflow_dist(wf_config)
    
    wf = Workflow(
        name=wf_config["name"],
        #parallelism=wf_config["parallelism"]
        )
    wf.add(dist_step)
    return wf

def submit_ft(
    wf_config
):
    """
    Major entry point for the whole workflow, only one config dict
    """
    # normalize args
    wf_config = normalize_args(wf_config)
    print(wf_config)

    global_config_workflow(wf_config)
    
    ft_step=workflow_finetune(wf_config)
    
    wf = Workflow(
        name=wf_config["name"],
        #parallelism=wf_config["parallelism"]
        )
    wf.add(ft_step)
    return wf




if __name__ == '__main__':
    import json
    with open('./input-abacus.json','r') as fp:
        config_dict=json.load(fp)
    submit_dist(config_dict)