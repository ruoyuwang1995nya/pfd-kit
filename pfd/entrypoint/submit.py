from copy import deepcopy
import os
from typing import Dict, List, Optional, Tuple
import json
import dpdata
import dflow
import dpgen2
import pfd
import re
import fpop
import ase

from dflow import ArgoStep, Step, Steps, Workflow, upload_artifact


from pfd.entrypoint.args import normalize as normalize_args
from pfd.entrypoint.common import global_config_workflow, expand_idx

from dpgen2.fp import fp_styles

from dpgen2.superop import PrepRunLmp, PrepRunDPTrain, PrepRunFp

from dpgen2.op import PrepLmp, RunLmp, PrepDPTrain, RunDPTrain

from pfd.flow.fine_tune import FineTune
from pfd.op import (
    PertGen,
    TaskGen,
    CollectData,
    Inference,
    SelectConfs,
    ModelTestOP,
    inference,
)

from pfd.superop import (
    ExplorationBlock,
    ExplFinetuneLoop,
    ExplFinetuneBlock,
    ExplDistBlock,
    ExplDistLoop,
)
from pfd.exploration.selector import (
    ConfFilters,
    ConfSelectorFrames,
    conf_filter_styles,
)

from pfd.exploration.render import TrajRenderLammps

from pfd.flow import Distillation
from pfd.constants import default_image
from dpgen2.utils.step_config import normalize as normalize_step_dict
from pfd.utils import (
    upload_artifact_and_print_uri,
    get_artifact_from_uri,
    matched_step_key,
    sort_slice_ops,
    print_keys_in_nice_format,
)
from periodictable import elements

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


def make_dist_op(
    prep_lmp_config: dict,
    run_lmp_config: dict,
    prep_train_config: dict,
    run_train_config: dict,
    gen_task_config: dict,
    collect_data_config: dict,
    pert_gen_config: dict,
    inference_config: dict,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    """
    Make a super OP template for distillation process
    """
    # build lmp op
    prep_run_lmp_op = PrepRunLmp(
        "prep-run-lmp-step",
        PrepLmp,
        RunLmp,
        prep_config=prep_lmp_config,
        run_config=run_lmp_config,
        upload_python_packages=upload_python_packages,
    )

    prep_run_dp_op = PrepRunDPTrain(
        "prep-run-dp-step",
        PrepDPTrain,
        RunDPTrain,
        prep_config=prep_train_config,
        run_config=run_train_config,
        upload_python_packages=upload_python_packages,
    )

    expl_dist_blk_op = ExplDistBlock(
        "expl-dist",
        gen_task_op=TaskGen,
        prep_run_explore_op=prep_run_lmp_op,
        prep_run_train_op=prep_run_dp_op,
        collect_data_op=CollectData,
        inference_op=Inference,
        gen_task_config=gen_task_config,
        inference_config=inference_config,
        collect_data_config=collect_data_config,
        upload_python_packages=upload_python_packages,
    )

    expl_dist_loop_op = ExplDistLoop(
        "expl-dist-loop", expl_dist_blk_op, upload_python_packages
    )

    dist_op = Distillation(
        "distillation",
        PertGen,
        expl_dist_loop_op,
        pert_gen_config,
        upload_python_packages=upload_python_packages,
    )
    return dist_op


def make_ft_op(
    fp_style: str = "vasp",
    # step configs
    pert_gen_step_config: dict = default_config,
    gen_task_step_config: dict = default_config,
    prep_fp_config: dict = default_config,
    run_fp_config: dict = default_config,
    prep_train_config: dict = default_config,
    run_train_config: dict = default_config,
    prep_lmp_config: dict = default_config,
    run_lmp_config: dict = default_config,
    collect_data_step_config: dict = default_config,
    select_confs_step_config: dict = default_config,
    inference_step_config: dict = default_config,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    init_training: bool = True,
    skip_aimd: bool = True,
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
    prep_run_train_dp_op = PrepRunDPTrain(
        "finetune",
        PrepDPTrain,
        RunDPTrain,
        prep_config=prep_train_config,
        run_config=run_train_config,
        upload_python_packages=upload_python_packages,
        # finetune=False,
        valid_data=None,
    )

    ## initiate LAMMPS op, possibly other engines
    prep_run_lmp_op = PrepRunLmp(
        "prep-run-lmp-step",
        PrepLmp,
        RunLmp,
        prep_config=prep_lmp_config,
        run_config=run_lmp_config,
        upload_python_packages=upload_python_packages,
    )

    expl_ft_blk_op = ExplFinetuneBlock(
        name="expl-ft-blk",
        gen_task_op=TaskGen,
        prep_run_explore_op=prep_run_lmp_op,
        prep_run_fp_op=prep_run_fp_op,
        collect_data_op=CollectData,
        select_confs_op=SelectConfs,
        prep_run_train_op=prep_run_train_dp_op,
        inference_op=ModelTestOP,
        collect_data_step_config=collect_data_step_config,
        inference_step_config=inference_step_config,
        select_confs_step_config=select_confs_step_config,
        gen_task_step_config=gen_task_step_config,
        upload_python_packages=upload_python_packages,
    )
    expl_finetune_loop_op = ExplFinetuneLoop(
        name="expl-ft-loop",
        expl_ft_blk_op=expl_ft_blk_op,
        upload_python_packages=upload_python_packages,
    )

    ft_op = FineTune(
        name="fine-tune",
        pert_gen_op=PertGen,
        prep_run_fp_op=prep_run_fp_op,
        collect_data_op=CollectData,
        prep_run_dp_train_op=prep_run_train_dp_op,
        expl_finetune_loop_op=expl_finetune_loop_op,
        pert_gen_step_config=pert_gen_step_config,
        collect_data_step_config=collect_data_step_config,
        upload_python_packages=upload_python_packages,
        init_training=init_training,
        skip_aimd=skip_aimd,
    )
    return ft_op


def get_systems_from_data(data, data_prefix=None):
    data = [data] if isinstance(data, str) else data
    assert isinstance(data, list)
    if data_prefix is not None:
        data = [os.path.join(data_prefix, ii) for ii in data]
    return data


def workflow_dist(
    config: Dict,
) -> Step:
    """
    Build a workflow from the OP templates of distillation
    """
    ## get input config
    default_step_config = config["default_step_config"]
    prep_train_step_config = config["step_configs"].get(
        "prep_train_config", default_step_config
    )
    run_train_step_config = config["step_configs"].get(
        "run_train_config", default_step_config
    )
    prep_lmp_step_config = config["step_configs"].get(
        "prep_explore_config", default_step_config
    )
    run_lmp_step_config = config["step_configs"].get(
        "run_explore_config", default_step_config
    )
    gen_lmp_step_config = config["step_configs"].get(
        "gen_lmp_config", default_step_config
    )
    collect_data_step_config = config["step_configs"].get(
        "collect_data_config", default_step_config
    )
    pert_gen_step_config = config["step_configs"].get(
        "pert_gen_config", default_step_config
    )
    inference_step_config = config["step_configs"].get(
        "inference_config", default_step_config
    )

    # uploaded python packages
    upload_python_packages = []
    custom_packages = config.get("upload_python_packages", [])
    upload_python_packages.extend(custom_packages)
    upload_python_packages.extend(list(dpdata.__path__))
    upload_python_packages.extend(list(dflow.__path__))
    upload_python_packages.extend(list(pfd.__path__))
    upload_python_packages.extend(list(dpgen2.__path__))
    ## task configs
    type_map = config["inputs"]["type_map"]
    if config["inputs"].get("mass_map") is not None:
        mass_map = config["inputs"]["mass_map"]
    else:
        mass_map = [getattr(elements, ii).mass for ii in type_map]
    pert_config = config["conf_generation"]
    # exploration
    explore_config = config["exploration"]["config"]
    expl_stages = config["exploration"]["stages"]
    converge_config = config["exploration"]["converge_config"]
    max_iter = config["exploration"].get("max_iter", 1)
    # train
    train_config = config["train"]["config"]
    # type_map_train=config["train"]["type_map"]
    numb_models = config["train"].get("numb_models", 1)
    # others

    collect_data_config = config["exploration"].get("test_set_config", {})
    collect_data_config["labeled_data"] = collect_data_config.get("labeled_data", False)
    collect_data_config["test_size"] = collect_data_config.get("test_size", 0.1)

    inference_config = {"task": "inference"}  # config["inference"]
    dp_test_config = deepcopy(inference_config)
    dp_test_config["task"] = "dp_test"

    ## prepare artifacts
    # read training template
    with open(config["train"]["template_script"], "r") as fp:
        template_script = json.load(fp)
    init_confs = upload_artifact(
        config["conf_generation"]["init_configurations"]["files"]
    )
    teacher_model = upload_artifact([config["inputs"]["teacher_model"]])

    # init_data
    if config["inputs"]["init_data_uri"] is not None:
        init_data = get_artifact_from_uri(config["inputs"]["init_data_uri"])
    elif config["inputs"]["init_data_sys"] is not None:
        init_data_prefix = config["inputs"]["init_data_prefix"]
        init_data = config["inputs"]["init_data_sys"]
        init_data = get_systems_from_data(init_data, init_data_prefix)
        init_data = upload_artifact_and_print_uri(init_data, "init_data")
    else:
        init_data = upload_artifact([])

    # make distillation op
    dist_op = make_dist_op(
        prep_lmp_step_config,
        run_lmp_step_config,
        prep_train_step_config,
        run_train_step_config,
        gen_lmp_step_config,
        collect_data_step_config,
        pert_gen_step_config,
        inference_step_config,
        upload_python_packages=upload_python_packages,
    )

    # make distillation steps
    dist_step = Step(
        "dp-dist-step",
        template=dist_op,
        parameters={
            "block_id": "dist",
            "type_map": type_map,
            "mass_map": mass_map,
            "pert_config": pert_config,
            "expl_stages": expl_stages,
            "numb_models": numb_models,
            "explore_config": explore_config,
            "converge_config": converge_config,
            "max_iter": max_iter,
            "template_script": template_script,
            "train_config": train_config,
            "inference_config": inference_config,
            "test_size": collect_data_config["test_size"],
            "type_map_train": [],  # type_map_train
        },
        artifacts={
            "init_confs": init_confs,
            "teacher_model": teacher_model,
            "init_data": init_data,
            "iter_data": upload_artifact([]),
        },
    )
    return dist_step


def workflow_finetune(config: Dict) -> Step:
    """
    Build a workflow from the OP templates of distillation
    """
    ## get input config
    fp_style = config["fp"]["type"]
    aimd_style = config["aimd"]["type"]
    default_config = config["default_step_config"]
    prep_train_config = config["step_configs"].get("prep_train_config", default_config)
    run_train_config = config["step_configs"].get("run_train_config", default_config)
    prep_fp_config = config["step_configs"].get("perp_fp_config", default_config)
    run_fp_config = config["step_configs"].get("run_fp_config", default_config)
    run_collect_data_config = config["step_configs"].get(
        "collect_data_config", default_config
    )
    run_select_confs_config = config["step_configs"].get(
        "select_confs_config", default_config
    )
    run_pert_gen_config = config["step_configs"].get("pert_gen_config", default_config)
    run_inference_config = config["step_configs"].get(
        "inference_config", default_config
    )
    prep_lmp_config = config["step_configs"].get("prep_explore_config", default_config)
    run_lmp_config = config["step_configs"].get("run_explore_config", default_config)
    # uploaded python packages
    upload_python_packages = []
    custom_packages = config.get("upload_python_packages", [])
    upload_python_packages.extend(custom_packages)
    upload_python_packages.extend(list(dpdata.__path__))
    upload_python_packages.extend(list(dflow.__path__))
    upload_python_packages.extend(list(pfd.__path__))
    upload_python_packages.extend(list(dpgen2.__path__))
    upload_python_packages.extend(list(fpop.__path__))
    upload_python_packages.extend(list(ase.__path__))
    ## task configs
    type_map = config["inputs"]["type_map"]
    if config["inputs"].get("mass_map") is not None:
        mass_map = config["inputs"]["mass_map"]
    else:
        mass_map = [getattr(elements, ii).mass for ii in type_map]
    train_config = config["train"]["config"]
    numb_models = config["train"].get("numb_models", 1)
    pert_config = config["conf_generation"]
    explore_config = config["exploration"]["md"]["config"]
    max_iter = config["exploration"]["md"].get("max_iter", 1)
    converge_config = config["exploration"]["converge_config"]
    # conf selectors
    conf_filters = get_conf_filters(config["exploration"]["filter"])
    render = TrajRenderLammps(nopbc=False)
    conf_selector = ConfSelectorFrames(render, config["fp"]["task_max"], conf_filters)

    expl_stages = config["exploration"]["md"]["stages"]
    init_training = config["exploration"].get("init_training", False)
    skip_aimd = config["exploration"].get("skip_aimd", True)
    if skip_aimd is True:
        print("AIMD is exploration skipped!")

    collect_data_config = {}
    collect_data_config["test_size"] = config["conf_generation"].get("test_data", 0.05)
    collect_data_config["system_partition"] = config["conf_generation"].get(
        "system_partition", False
    )
    collect_data_config["labeled_data"] = True

    ## prepare artifacts
    # read training template
    with open(config["train"]["template_script"], "r") as fp:
        template_script = json.load(fp)
    init_confs = upload_artifact(
        config["conf_generation"]["init_configurations"]["files"]
    )

    # init_data
    if config["inputs"]["init_data_uri"] is not None:
        init_data = get_artifact_from_uri(config["inputs"]["init_data_uri"])
    elif config["inputs"]["init_data_sys"] is not None:
        init_data_prefix = config["inputs"]["init_data_prefix"]
        init_data = config["inputs"]["init_data_sys"]
        init_data = get_systems_from_data(init_data, init_data_prefix)
        init_data = upload_artifact_and_print_uri(init_data, "init_data")
    else:
        init_data = upload_artifact([])
    iter_data = upload_artifact([])
    init_models_paths = config["train"].get("init_models_paths", None)
    if config["train"].get("init_models_url") is not None:
        print("Using uploaded model at: ", config["train"].get("init_models_url"))
        init_models = get_artifact_from_uri(config["train"]["init_models_url"])
    elif init_models_paths is not None:
        init_models = upload_artifact_and_print_uri(init_models_paths, "init_models")
    else:
        raise FileNotFoundError("Pre-trained model must exist!")

    fp_config = {}
    fp_inputs_config = config["fp"]["inputs_config"]
    fp_type = config["fp"]["type"]
    fp_inputs = fp_styles[fp_style]["inputs"](**fp_inputs_config)
    fp_config["inputs"] = fp_inputs
    fp_config["run"] = config["fp"]["run_config"]
    fp_config["extra_output_files"] = config["fp"].get("extra_output_files", [])

    aimd_config = {}
    aimd_inputs_config = config["aimd"]["inputs_config"]
    aimd_type = config["aimd"]["type"]
    aimd_inputs = fp_styles[aimd_style]["inputs"](**aimd_inputs_config)
    aimd_config["inputs"] = aimd_inputs
    aimd_config["run"] = config["aimd"]["run_config"]
    aimd_config["extra_output_files"] = config["aimd"].get("extra_output_files", [])

    # make distillation op
    ft_op = make_ft_op(
        fp_style=fp_type,
        pert_gen_step_config=run_pert_gen_config,
        prep_fp_config=prep_fp_config,
        run_fp_config=run_fp_config,
        prep_train_config=prep_train_config,
        run_train_config=run_train_config,
        prep_lmp_config=prep_lmp_config,
        run_lmp_config=run_lmp_config,
        collect_data_step_config=run_collect_data_config,
        select_confs_step_config=run_select_confs_config,
        inference_step_config=run_inference_config,
        upload_python_packages=upload_python_packages,
        init_training=init_training,
        skip_aimd=skip_aimd,
    )

    ft_step = Step(
        "fine-tune",
        template=ft_op,
        parameters={
            "block_id": "finetune",
            "type_map": type_map,
            "mass_map": mass_map,
            "pert_config": pert_config,  # Total input parameter file: to be changed in the future
            "numb_models": numb_models,
            "expl_stages": expl_stages,
            "conf_selector": conf_selector,
            "converge_config": converge_config,
            "max_iter": max_iter,
            "explore_config": explore_config,
            "template_script": template_script,
            "train_config": train_config,
            "fp_config": fp_config,
            "aimd_config": aimd_config,
            "collect_data_config": collect_data_config,
        },
        artifacts={
            "init_models": init_models,
            "init_confs": init_confs,
            "expl_models": init_models,
            "init_data": init_data,
            "iter_data": iter_data,
        },
    )
    return ft_step


def submit_dist(
    wf_config, reuse_step: Optional[List[ArgoStep]] = None, no_submission: bool = False
):
    """
    Major entry point for the whole workflow, only one config dict
    """
    # normalize args
    wf_config = normalize_args(wf_config)
    global_config_workflow(wf_config)
    dist_step = workflow_dist(wf_config)
    wf = Workflow(
        name=wf_config["name"],
        # parallelism=wf_config["parallelism"]
    )
    wf.add(dist_step)

    if not no_submission:
        wf.submit(reuse_step=reuse_step)
    return wf


def submit_ft(
    wf_config,
    reuse_step: Optional[List[ArgoStep]] = None,
    no_submission: bool = False,
):
    """
    Major entry point for the whole workflow, only one config dict
    """
    # normalize args
    wf_config = normalize_args(wf_config)
    global_config_workflow(wf_config)
    ft_step = workflow_finetune(wf_config)

    wf = Workflow(
        name=wf_config["name"],
        # parallelism=wf_config["parallelism"]
    )
    wf.add(ft_step)
    if not no_submission:
        wf.submit(reuse_step=reuse_step)
    return wf


def successful_step_keys(wf):
    all_step_keys = []
    steps = wf.query_step()
    # For reused steps whose startedAt are identical, sort them by key
    steps.sort(key=lambda x: "%s-%s" % (x.startedAt, x.key))
    for step in steps:
        if step.key is not None and step.phase == "Succeeded":
            all_step_keys.append(step.key)
    return all_step_keys


def get_superop(key):
    if "prep-train" in key:
        return key.replace("prep-train", "prep-run-train")
    elif "run-train-" in key:
        return re.sub("run-train-[0-9]*", "prep-run-train", key)
    elif "prep-lmp" in key:
        return key.replace("prep-lmp", "prep-run-explore")
    elif "run-lmp-" in key:
        return re.sub("run-lmp-[0-9]*", "prep-run-explore", key)
    elif "prep-fp" in key:
        return key.replace("prep-fp", "prep-run-fp")
    elif "run-fp-" in key:
        return re.sub("run-fp-[0-9]*", "prep-run-fp", key)
    elif "prep-caly-input" in key:
        return key.replace("prep-caly-input", "prep-run-explore")
    elif "collect-run-calypso-" in key:
        return re.sub("collect-run-calypso-[0-9]*-[0-9]*", "prep-run-explore", key)
    elif "prep-dp-optim-" in key:
        return re.sub("prep-dp-optim-[0-9]*-[0-9]*", "prep-run-explore", key)
    elif "run-dp-optim-" in key:
        return re.sub("run-dp-optim-[0-9]*-[0-9]*-[0-9]*", "prep-run-explore", key)
    elif "prep-caly-model-devi" in key:
        return key.replace("prep-caly-model-devi", "prep-run-explore")
    elif "run-caly-model-devi" in key:
        return re.sub("run-caly-model-devi-[0-9]*", "prep-run-explore", key)
    elif "caly-evo-step" in key:
        return re.sub("caly-evo-step-[0-9]*", "prep-run-explore", key)
    return None


def fold_keys(all_step_keys):
    folded_keys = {}
    for key in all_step_keys:
        is_superop = False
        for superop in ["prep-run-train", "prep-run-explore", "prep-run-fp"]:
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


def get_resubmit_keys(
    wf,
):
    all_step_keys = successful_step_keys(wf)

    step_keys = [
        "pert-gen",
        "prep-run-train",
        "prep-train",
        "run-train",
        "modify-train-script",
        "prep-caly-input",
        "prep-caly-model-devi",
        "run-caly-model-devi",
        "prep-run-explore",
        "prep-lmp",
        "run-lmp",
        "select-confs",
        "prep-run-fp",
        "prep-fp",
        "run-fp",
        "collect-data",
        "validation-test",
        "scheduler",
        "id",
    ]

    all_step_keys = matched_step_key(
        all_step_keys,
        step_keys,
    )
    all_step_keys = sort_slice_ops(
        all_step_keys,
        ["run-train", "run-lmp", "run-fp"],
    )
    folded_keys = fold_keys(all_step_keys)
    return folded_keys


def resubmit_workflow(
    wf_config,
    wfid,
    list_steps=False,
    reuse=None,
    fold=False,
    flow_type="dist",
    **kwargs,
):
    wf_config = normalize_args(wf_config)
    global_config_workflow(wf_config)
    old_wf = Workflow(id=wfid)
    folded_keys = get_resubmit_keys(old_wf)
    all_step_keys = sum(folded_keys.values(), [])

    if list_steps:
        prt_str = print_keys_in_nice_format(
            all_step_keys,
            ["run-train", "run-lmp", "run-fp"],
        )
        print(prt_str)

    if reuse is None:
        return None
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

    if flow_type == "dist":
        return submit_dist(wf_config, reuse_step=reuse_step, **kwargs)
    elif flow_type == "finetune":
        return submit_ft(wf_config, reuse_step=reuse_step, **kwargs)

    else:
        raise NotImplementedError("%d has not been implemented!" % (flow_type))


if __name__ == "__main__":
    import json

    with open("./input-abacus.json", "r") as fp:
        config_dict = json.load(fp)
    submit_dist(config_dict)
