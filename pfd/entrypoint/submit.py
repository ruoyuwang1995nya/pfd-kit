from copy import deepcopy
import os
import copy
from turtle import up
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import dpdata
import dflow
import pfd
import re
import ase
import warnings
import time
from dflow import ArgoStep, Step, Steps, Workflow, upload_artifact, download_artifact
from pfd import train
from pfd.entrypoint.args import (
    normalize as normalize_args,
)
from pfd.entrypoint.common import global_config_workflow, expand_idx

from pfd.exploration.task.stage import ExplorationStage
from pfd.train import train_styles
#from pfd.exploration import explore_styles
from pfd.fp import fp_styles
from pfd.flow import wf_styles
from pfd.superop import PrepRunFp

from pfd.op import (
    CollectData,
    SelectConfs,
    ModelTestOP,
)

from pfd.flow import (
    ExplTrainBlock,
    ExplTrainLoop,
    PFD
)

from pfd.exploration.selector import (
    ConfFilters,
    ConfSelectorFrames,
    conf_filter_styles,
)
from pfd.exploration.converge import ConfFiltersConv, ConfFilterConv
from pfd.exploration.render import TrajRender
from pfd.exploration.scheduler import Scheduler
from pfd.constants import default_image
from pfd.utils.step_config import normalize as normalize_step_dict
from pfd.utils import (
    upload_artifact_and_print_uri,
    get_artifact_from_uri,
    matched_step_key,
    sort_slice_ops,
    print_keys_in_nice_format,
)

from pfd.superop import PrepRunExpl, PrepRunCaly
from pfd.op import (
    PrepASE,
    RunASE,
    PrepCalyInput,
    PrepCalyASEOptim,
    RunCalyASEOptim,
    CollRunCaly,
    CalyEvoStep,
    CalyEvoStepMerge,
)
from pfd.exploration.task import (
    AseTaskGroup, CalyTaskGroup
)


explore_styles = {
    "ase":{
        "preprun": PrepRunExpl,
        "prep": PrepASE,
        "run": RunASE,
        "task_grp": AseTaskGroup,
    },
    "calypso":{
        "preprun": PrepRunCaly,
        "prep": PrepCalyInput,
        "run": CalyEvoStep,
        "task_grp": CalyTaskGroup,
    }
}




default_config = normalize_step_dict({"template_config": {"image": default_image}})


def get_conf_filters(config):
    conf_filters = None
    if len(config) > 0:
        conf_filters = ConfFilters()
        for c in config:
            c = deepcopy(c)
            conf_filter = conf_filter_styles[c.pop("type")](**c)
            conf_filters.add(conf_filter)
    return conf_filters


def get_conf_filters_conv(config):
    conf_filters_conv = ConfFiltersConv()
    if len(config) > 0:
        for c in config:
            c = deepcopy(c)
            conf_filter_conv = ConfFilterConv.get_filter(c.pop("type"))(**c)
            conf_filters_conv.add(conf_filter_conv)
    return conf_filters_conv

def make_pfd_op(
    fp_style: str = "vasp",
    train_style: str = "dp",
    explore_style: str = "ase",
    wf_style: str = "finetune",
    #pert_gen_step_config: dict = default_config,
    prep_fp_config: dict = default_config,
    run_fp_config: dict = default_config,
    train_config: dict = default_config,
    #run_train_config: dict = default_config,
    prep_explore_config: dict = default_config,
    run_explore_config: dict = default_config,
    scheduler_config: dict = default_config,
    collect_data_config: dict = default_config,
    select_confs_config: dict = default_config,
    evaluate_config: dict = default_config,
    #inference_step_config: dict = default_config,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    init_train: bool = False,
    init_fp: bool = False,
):
    ## initiate fp op
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

    ## initiate DP train op
    if train_style in train_styles.keys():
        train_op = train_styles[train_style]
    else:
        raise NotImplementedError(
            f"Training style {train_style} has not been implemented!"
        )

    if explore_style == "ase":
        prep_run_explore_op = explore_styles[explore_style]["preprun"](
            "prep-run-explore-step",
            explore_styles[explore_style]["prep"],
            explore_styles[explore_style]["run"],
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
    elif explore_style in ["calypso","calypso:merge"]:
        if explore_style == "calypso":
            caly_evo_step_op = CalyEvoStep(
                name="caly-eval-step",
                collect_run_caly=CollRunCaly,
                prep_ase_optim=PrepCalyASEOptim,
                run_ase_optim=RunCalyASEOptim,
                prep_config=prep_explore_config,
                run_config=run_explore_config,
            )
        else:
            caly_evo_step_op = CalyEvoStepMerge(
            name="caly-eval-step",
            collect_run_caly=CollRunCaly,
            prep_ase_optim=PrepCalyASEOptim,
            run_ase_optim=RunCalyASEOptim,
            prep_config=prep_explore_config,
            run_config=run_explore_config,
    )
        prep_run_explore_op = PrepRunCaly(
            "prep-run-caly-step",
            PrepCalyInput,
            caly_evo_step_op,
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
    else:
        raise ValueError(f"Explore style {explore_style} has not been implemented!")

    expl_train_block_op = ExplTrainBlock(
        name="blk",
        prep_run_explore_op=prep_run_explore_op,
        prep_run_fp_op=prep_run_fp_op,
        collect_data_op=CollectData,
        select_confs_op=SelectConfs,
        prep_run_train_op=train_op,
        evaluate_op= ModelTestOP,
        train_config=train_config,
        collect_data_config=collect_data_config,
        select_confs_config=select_confs_config,
        evaluate_config=evaluate_config,
        upload_python_packages=upload_python_packages,
    )
    
    expl_train_loop_op = ExplTrainLoop(
        name="loop",
        expl_train_blk_op=expl_train_block_op,
        stage_scheduler_op=wf_styles[wf_style], ## implement a default scheduler
        scheduler_config=scheduler_config,
        upload_python_packages=upload_python_packages,
    )

    pfd_op = PFD(
        name="pfd",
        prep_run_fp_op=prep_run_fp_op,
        collect_data_op=CollectData,
        train_op=train_op,
        expl_train_loop_op=expl_train_loop_op,
        train_config=train_config,
        collect_data_step_config=collect_data_config,
        upload_python_packages=upload_python_packages,
        init_train=init_train,
        init_fp=init_fp,
    )
    return pfd_op

def get_systems_from_data(data, data_prefix=None):
    data = [data] if isinstance(data, str) else data
    assert isinstance(data, list)
    if data_prefix is not None:
        data = [os.path.join(data_prefix, ii) for ii in data]
    return data


class FlowGen:
    def __init__(
        self,
        config: Dict,
        debug: bool = False,
        download_path: Union[Path, str] = Path("./"),
    ):
        self._download_path = download_path
        if debug is True:
            os.environ["DFLOW_DEBUG"] = "1"
        elif os.environ.get("DFLOW_DEBUG"):
            del os.environ["DFLOW_DEBUG"]
        self._config = normalize_args(config)
        global_config_workflow(self._config)
        print("dflow mode: %s" % dflow.config["mode"])
        self.workflow = Workflow(name=self._config["name"])
        self._wf_type = config["task"].get("type")
        self._set_wf(self._config)

    @property
    def wf_type(self):
        return self._wf_type

    @property
    def download_path(self):
        if isinstance(self._download_path, str):
            return Path(self._download_path)
        else:
            return self._download_path


    def _set_wf(self, config: Dict):
        """
        Build a workflow from the OP templates of model finetune
        """
        # executor config start with prep_* or run_*
        default_config = config["default_step_config"]
        run_train_config = config["step_configs"].get(
            "run_train_config", default_config
        )
        prep_fp_config = config["step_configs"].get("perp_fp_config", default_config)
        run_fp_config = config["step_configs"].get("run_fp_config", default_config)
        run_collect_data_config = config["step_configs"].get(
            "collect_data_config", default_config
        )
        run_select_confs_config = config["step_configs"].get(
            "select_confs_config", default_config
        )
        prep_explore_config = config["step_configs"].get(
            "prep_explore_config", default_config
        )
        run_explore_config = config["step_configs"].get(
            "run_explore_config", default_config
        )
        run_scheduler_config = config["step_configs"].get(
            "run_scheduler_config", default_config
        )
        run_evaluate_config = config["step_configs"].get(
            "run_evaluate_config", default_config
        )

        # uploaded python packages
        upload_python_packages = []
        if custom_packages := config.get("upload_python_packages"):
            upload_python_packages.extend(custom_packages)
        upload_python_packages.extend(list(dpdata.__path__))
        upload_python_packages.extend(list(dflow.__path__))
        upload_python_packages.extend(list(pfd.__path__))
        upload_python_packages.extend(list(ase.__path__))
        
        
        ##### task configs
        task_type = config["task"].get("type", "finetune")
        max_iter = config["task"]["max_iter"]
        
        if task_type == 'dist':
            init_train=False
            init_fp=False
            if max_iter > 1:
                warnings.warn(
        "In most cases, there is absolutely no need for more than one training iteration for knowledge distillation!"
    )
        elif task_type == 'finetune':
            init_train = config["task"]["init_train"]
            if init_train is False:
                print("No initial training before exploration")
            init_fp = config["task"]["init_fp"]
            if init_fp is False:
                print("Initial fp calculation skipped")

        
        #### train config
        train_style = config["train"]["type"]
        train_config = config["train"]["config"]
        if task_type == 'finetune':
            train_config["finetune_mode"] = True
        else:
            train_config["finetune_mode"] = False
        # read custom training template
        if isinstance(config["train"]["template_script"], str):
            with open(config["train"]["template_script"], "r") as fp:
                template_script = json.load(fp)
        elif isinstance(config["train"]["template_script"], dict):
            template_script = config["train"]["template_script"]

        #### explore config
        explore_style = config["exploration"]["type"]
        expl_stages = config["exploration"]["stages"]
        explore_config = config["exploration"]["config"]

        #### confs selection config
        select_confs_config = config["select_confs"]
        conf_filters = get_conf_filters(select_confs_config.pop("frame_filter"))
        render = TrajRender.get_driver(explore_style)()
        conf_selector = ConfSelectorFrames(
            render,  conf_filters=conf_filters
        )
        
        #### model evaluation config
        evaluate_config = config["evaluate"]

        ##### collect_data_config 
        collect_data_config = {"test_size": evaluate_config.pop('test_size',0.1)}

        #### read init confs
        if config["inputs"]["init_confs"]["confs_paths"] is not None:
            init_confs_prefix = config["inputs"]["init_confs"]["prefix"]
            init_confs = config["inputs"]["init_confs"]["confs_paths"]
            init_confs = get_systems_from_data(init_confs, init_confs_prefix)
        else:
            raise RuntimeError("init_confs must be provided")
        
        #### read init fp confs
        if config["inputs"]["init_fp_confs"]["confs_paths"] is not None:
            init_fp_confs_prefix = config["inputs"]["init_fp_confs"]["prefix"]
            init_fp_confs = config["inputs"]["init_fp_confs"]["confs_paths"]
            init_fp_confs = get_systems_from_data(init_fp_confs, init_fp_confs_prefix)
        else: 
            init_fp_confs = []
        init_fp_confs = upload_artifact_and_print_uri(init_fp_confs, "init_fp_confs")

        #### create expl tasks from the init_confs
        task_grp_style = explore_styles[explore_style]["task_grp"]
        _expl_stages=[]
        for stg in expl_stages:
            expl_stage=ExplorationStage()
            for task_grp in stg:
                # Use the unified method for creating task groups
                expl_stage.add_task_group(
                    task_grp_style.make_task_grp_from_conf(
                        task_grp,
                        init_confs,
                    )
                )
            _expl_stages.append(expl_stage)

        #### scheduler for workflow management
        scheduler = Scheduler(
            explore_stages=_expl_stages,
            max_iter=max_iter,
            train_config=train_config
        )

        #### upload init_data
        if config["inputs"]["init_data_uri"] is not None:
            init_data = get_artifact_from_uri(config["inputs"]["init_data_uri"])
        elif config["inputs"]["init_data_sys"] is not None:
            init_data_prefix = config["inputs"]["init_data_prefix"]
            init_data = config["inputs"]["init_data_sys"]
            init_data = get_systems_from_data(init_data, init_data_prefix)
            init_data = upload_artifact_and_print_uri(init_data, "init_data")
        else:
            init_train = init_fp
            init_data = upload_artifact([])
        
        #### upload init models
        init_model_paths = config["inputs"]["base_model_path"]
        if config["inputs"]["base_model_uri"] is not None:
            print("Using uploaded model at: ", config["inputs"]["base_model_uri"])
            init_model = get_artifact_from_uri(config["inputs"]["base_model_uri"])
        elif init_model_paths:
            init_model = upload_artifact_and_print_uri(init_model_paths, "base_model")
        else:
            raise FileNotFoundError("Pre-trained model must exist!")

        fp_config = {}
        fp_inputs_config = config["fp"]["inputs_config"]
        fp_type = config["fp"]["type"]
        if task_type == 'dist':
            fp_type = 'ase'
        fp_inputs = fp_styles[fp_type]["inputs"](**fp_inputs_config)
        fp_config["inputs"] = fp_inputs
        fp_config["run"] = config["fp"]["run_config"]
        fp_config["extra_output_files"] = config["fp"].get("extra_output_files", [])

        # aimd exploration
        init_fp_config = {}
        if init_fp:
            init_fp_config.update(fp_config)
            init_fp_inputs_config = copy.deepcopy(fp_inputs_config)
            init_fp_inputs_config.update(config["init_fp"]["inputs_config"])
            init_fp_inputs = fp_styles[fp_type]["inputs"](**init_fp_inputs_config)
            init_fp_config["inputs"] = init_fp_inputs

        # make pfd op
        pfd_op = make_pfd_op(
            wf_style=task_type,
            fp_style=fp_type,
            train_style=train_style,
            explore_style=explore_style,
            prep_fp_config=prep_fp_config,
            run_fp_config=run_fp_config,
            train_config=run_train_config,
            prep_explore_config=prep_explore_config,
            run_explore_config=run_explore_config,
            scheduler_config=run_scheduler_config,
            collect_data_config=run_collect_data_config,
            select_confs_config=run_select_confs_config,
            evaluate_config=run_evaluate_config,
            upload_python_packages=upload_python_packages,
            init_train=init_train,
            init_fp=init_fp,
        )
        
        pfd_step = Step(
            "workflow",
            template=pfd_op,
            parameters={
                "block_id": "finetune",
                
                "explore_config": explore_config,
                "conf_selector": conf_selector,
                "select_confs_config": select_confs_config,
                "scheduler": scheduler,
                
                # training
                "template_script": template_script,
                "train_config":train_config,
                
                # fp_calculation
                "fp_config": fp_config,
                "init_fp_config":init_fp_config,
                
                #"aimd_config": aimd_config,
                #"aimd_sample_conf": aimd_sample_conf,
                "collect_data_config": collect_data_config,
                "evaluate_config": evaluate_config,
            },
            artifacts={
                "init_model": init_model,
                "expl_model": init_model,
                "init_data": init_data,
                "init_confs": init_confs,
                "init_fp_confs": init_fp_confs
                #"iter_data": iter_data,
            },
        )
        self.workflow.add(pfd_step)

    def _moniter(self):
        while True:
            time.sleep(4)
            step_info = self.workflow.query()
            wf_status = self.workflow.query_status()
            if wf_status == "Failed":
                raise RuntimeError(
                    f"Workflow failed (ID: {self.workflow.id}, UID: {self.workflow.uid})"
                )
            try:
                dist_post = step_info.get_step(name="finetune")[0]
            except IndexError:
                continue
            if dist_post["phase"] == "Succeeded":
                print(
                    f"Distillation finished (ID: {self.workflow.id}, UID: {self.workflow.uid})"
                )
                print("Retrieving completed tasks to local...")
                download_artifact(
                    artifact=dist_post.outputs.artifacts["model"],
                    path=self.download_path,
                )
                break

    def submit(
        self,
        reuse_step: Optional[List[ArgoStep]] = None,
        no_submission: bool = False,
        only_submit: bool = True,
    ):
        if not no_submission:
            self.workflow.submit(reuse_step=reuse_step)
        else:
            return self.workflow
        if not only_submit:
            self._moniter()

def successful_step_keys(wf, unsuccessful_step_keys: bool = False):
    """[From DPGEN2] Get the keys of all successful steps in the workflow.

    Args:
        wf (_type_): The workflow object.
        unsuccessful_step_keys (bool, optional): If True, include keys of unsuccessful steps. Defaults to False.

    Returns:
        list: A list of successful step keys.
    """
    all_step_keys = []
    steps = wf.query_step()
    # For reused steps whose startedAt are identical, sort them by key
    steps.sort(key=lambda x: "%s-%s" % (x.startedAt, x.key))
    for step in steps:
        if not unsuccessful_step_keys:
            if step.key is not None and step.phase == "Succeeded":
                all_step_keys.append(step.key)
        else:
            if step.key is not None:
                all_step_keys.append(step.key)
    return all_step_keys


def get_superop(key):
    """[From DPGEN2] Get the super operation key for a given step key.

    Args:
        key (str): The step key.

    Returns:
        str: The super operation key, or None if not found.
    """
    if "prep-expl" in key:
        return key.replace("prep-expl", "prep-run-explore")
    elif "run-expl-" in key:
        return re.sub("run-expl-[0-9]*", "prep-run-explore", key)
    elif "prep-fp" in key:
        return key.replace("prep-fp", "prep-run-fp")
    elif "run-fp-" in key:
        return re.sub("run-fp-[0-9]*", "prep-run-fp", key)
    return None


def fold_keys(all_step_keys):
    folded_keys = {}
    for key in all_step_keys:
        is_superop = False
        for superop in ["prep-run-explore", "prep-run-fp"]:
            if superop in key:
                if key not in folded_keys:
                    folded_keys[key] = []
                is_superop = True
                break
        if is_superop:
            continue
        superop = get_superop(key)
        # if its super OP is succeeded, fold it into its super OP
        if superop is not None and superop in all_step_keys:
            if superop not in folded_keys:
                folded_keys[superop] = []
            folded_keys[superop].append(key)
        else:
            folded_keys[key] = [key]
    for k, v in folded_keys.items():
        if v == []:
            folded_keys[k] = [k]
    return folded_keys


def get_resubmit_keys(wf, unsuccessful_step_keys: bool = False):
    """[From DPGEN2] Get the keys of all steps in the workflow for resubmission.
    """
    all_step_keys = successful_step_keys(wf, unsuccessful_step_keys)
    # legal step keys
    step_keys = [
        "train",
        "prep-run-explore",
        "prep-expl",
        "run-expl",
        "select-confs",
        "prep-run-fp",
        "prep-fp",
        "run-fp",
        "collect-data",
        "evaluate",
        # "scheduler",
    ]

    all_step_keys = matched_step_key(
        all_step_keys,
        step_keys,
    )
    all_step_keys = sort_slice_ops(
        all_step_keys,
        ["run-expl", "run-fp"],
    )
    folded_keys = fold_keys(all_step_keys)
    return folded_keys


def resubmit_workflow(
    wf_config,
    wfid,
    list_steps=False,
    reuse=None,
    fold=False,
    unsuccessful_step_keys: bool = False,
    **kwargs,
):
    global_config_workflow(normalize_args(wf_config))
    old_wf = Workflow(id=wfid)
    folded_keys = get_resubmit_keys(old_wf, unsuccessful_step_keys)
    all_step_keys = sum(folded_keys.values(), [])
    if list_steps:
        prt_str = print_keys_in_nice_format(
            all_step_keys, ["run-train", "run-lmp", "run-fp"]
        )
        print(prt_str)
        return
    if reuse is None:
        return
    reuse_idx = expand_idx(reuse)
    reused_keys = [all_step_keys[ii] for ii in reuse_idx]
    if fold:
        reused_folded_keys = {}
        for key in reused_keys:
            superop = get_superop(key)
            if superop is not None:
                if superop not in reused_folded_keys:
                    reused_folded_keys[superop] = []
                reused_folded_keys[superop].append(key)
            else:
                reused_folded_keys[key] = [key]
        for k, v in reused_folded_keys.items():
            # reuse the super OP iif all steps within it are reused
            if v != [k] and k in folded_keys and set(v) == set(folded_keys[k]):
                reused_folded_keys[k] = [k]
        reused_keys = sum(reused_folded_keys.values(), [])
        print(reused_keys)
    reuse_step = old_wf.query_step(key=reused_keys)
    # For reused steps whose startedAt are identical, sort them by key
    reuse_step.sort(key=lambda x: "%s-%s" % (x.startedAt, x.key))
    wf = FlowGen(wf_config)
    wf.submit(reuse_step=reuse_step, **kwargs)
