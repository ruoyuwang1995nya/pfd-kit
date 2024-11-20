from copy import deepcopy
import os
import copy
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import dpdata
import dflow
import dpgen2
import pfd
import re
import fpop
import ase
import time
from dflow import ArgoStep, Step, Steps, Workflow, upload_artifact, download_artifact


from pfd.entrypoint.args import normalize_infer_args, normalize as normalize_args
from pfd.entrypoint.common import global_config_workflow, expand_idx

from dpgen2.fp import fp_styles
from pfd.train import train_styles
from pfd.exploration import explore_styles

from dpgen2.superop import PrepRunDPTrain, PrepRunFp

from pfd.op import (
    PertGen,
    CollectData,
    InferenceOP,
    SelectConfs,
    ModelTestOP,
)

from pfd.superop import (
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
from pfd.exploration.converge import ConfFiltersConv, ConfFilterConv
from pfd.exploration.render import TrajRenderLammps
from pfd.exploration.scheduler import Scheduler
from pfd.flow import Distillation, FineTune, DataGen
from pfd.constants import default_image
from pfd.utils.step_config import normalize as normalize_step_dict
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


def get_conf_filters_conv(config):
    conf_filters_conv = ConfFiltersConv()
    if len(config) > 0:
        for c in config:
            c = deepcopy(c)
            conf_filter_conv = ConfFilterConv.get_filter(c.pop("type"))(**c)
            conf_filters_conv.add(conf_filter_conv)
    return conf_filters_conv


def make_dist_op(
    teacher_model_style: str = "dp",
    model_style: str = "dp",
    explore_style: str = "lmp",
    prep_lmp_config: dict = default_config,
    run_lmp_config: dict = default_config,
    prep_train_config: dict = default_config,
    run_train_config: dict = default_config,
    scheduler_config: dict = default_config,
    collect_data_config: dict = default_config,
    pert_gen_config: dict = default_config,
    inference_config: dict = default_config,
    model_test_config: dict = default_config,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    """
    Make a super OP template for distillation process
    """
    # build lmp op
    if teacher_model_style in explore_styles.keys():
        if explore_style in explore_styles[teacher_model_style].keys():
            prep_run_lmp_op = explore_styles[teacher_model_style][explore_style][
                "preprun"
            ](
                "prep-run-explore-step",
                explore_styles[teacher_model_style][explore_style]["prep"],
                explore_styles[teacher_model_style][explore_style]["run"],
                prep_config=prep_lmp_config,
                run_config=run_lmp_config,
                upload_python_packages=upload_python_packages,
            )
        else:
            raise NotImplementedError(
                f"Explore style {explore_style} has not been implemented!"
            )

    else:
        raise NotImplementedError(
            f"Model style {teacher_model_style} has not been implemented!"
        )

    ## initiate DP train op
    if model_style in train_styles.keys():
        prep_run_train_op = PrepRunDPTrain(
            "finetune",
            train_styles[model_style]["prep"],
            train_styles[model_style]["run"],
            prep_config=prep_train_config,
            run_config=run_train_config,
            upload_python_packages=upload_python_packages,
            valid_data=None,
        )
    else:
        raise NotImplementedError(
            f"Training style {model_style} has not been implemented!"
        )

    expl_dist_blk_op = ExplDistBlock(
        "expl-dist",
        prep_run_explore_op=prep_run_lmp_op,
        prep_run_train_op=prep_run_train_op,
        collect_data_op=CollectData,
        inference_op=InferenceOP,
        inference_config=inference_config,
        model_test_config=model_test_config,
        collect_data_config=collect_data_config,
        upload_python_packages=upload_python_packages,
    )

    expl_dist_loop_op = ExplDistLoop(
        "expl-dist-loop", expl_dist_blk_op, scheduler_config, upload_python_packages
    )
    dist_op = Distillation(
        "distillation",
        PertGen,
        expl_dist_loop_op,
        pert_gen_config,
        upload_python_packages=upload_python_packages,
    )
    return dist_op


def make_data_gen_op(
    fp_style: str = "vasp",
    prep_fp_config: dict = default_config,
    run_fp_config: dict = default_config,
    pert_gen_config: dict = default_config,
    collect_data_config: dict = default_config,
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    """
    Creates a DataGen operation.

    Args:
        fp_style (str): The style of the force field calculation (default is "vasp").
        prep_fp_config (dict): Configuration for preparing the force field calculation.
        run_fp_config (dict): Configuration for running the force field calculation.
        pert_gen_config (dict): Configuration for perturbation generation.
        collect_data_config (dict): Configuration for data collection.
        upload_python_packages (Optional[List[os.PathLike]]): List of Python packages to upload.

    Returns:
        DataGen: An instance of the DataGen class.
    """

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

    return DataGen(
        "data-gen",
        PertGen,
        prep_run_fp_op,
        CollectData,
        pert_gen_config,
        collect_data_config,
        upload_python_packages,
    )


def make_ft_op(
    fp_style: str = "vasp",
    train_style: str = "dp",
    explore_style: str = "lmp",
    pert_gen_step_config: dict = default_config,
    prep_fp_config: dict = default_config,
    run_fp_config: dict = default_config,
    prep_train_config: dict = default_config,
    run_train_config: dict = default_config,
    prep_lmp_config: dict = default_config,
    run_lmp_config: dict = default_config,
    scheduler_config: dict = default_config,
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
    if train_style in train_styles.keys():
        prep_run_train_op = PrepRunDPTrain(
            "finetune",
            train_styles[train_style]["prep"],
            train_styles[train_style]["run"],
            prep_config=prep_train_config,
            run_config=run_train_config,
            upload_python_packages=upload_python_packages,
            valid_data=None,
        )
    else:
        raise NotImplementedError(
            f"Training style {train_style} has not been implemented!"
        )

    if explore_style in explore_styles[train_style].keys():
        prep_run_lmp_op = explore_styles[train_style][explore_style]["preprun"](
            "prep-run-explore-step",
            explore_styles[train_style][explore_style]["prep"],
            explore_styles[train_style][explore_style]["run"],
            prep_config=prep_lmp_config,
            run_config=run_lmp_config,
            upload_python_packages=upload_python_packages,
        )
    else:
        raise ValueError(f"Explore style {explore_style} has not been implemented!")

    expl_ft_blk_op = ExplFinetuneBlock(
        name="expl-ft-blk",
        prep_run_explore_op=prep_run_lmp_op,
        prep_run_fp_op=prep_run_fp_op,
        collect_data_op=CollectData,
        select_confs_op=SelectConfs,
        prep_run_train_op=prep_run_train_op,
        inference_op=ModelTestOP,
        collect_data_step_config=collect_data_step_config,
        inference_step_config=inference_step_config,
        select_confs_step_config=select_confs_step_config,
        upload_python_packages=upload_python_packages,
    )
    expl_finetune_loop_op = ExplFinetuneLoop(
        name="expl-ft-loop",
        expl_ft_blk_op=expl_ft_blk_op,
        scheduler_config=scheduler_config,
        upload_python_packages=upload_python_packages,
    )

    ft_op = FineTune(
        name="fine-tune",
        pert_gen_op=PertGen,
        prep_run_fp_op=prep_run_fp_op,
        collect_data_op=CollectData,
        prep_run_dp_train_op=prep_run_train_op,
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
        if self._wf_type == "dist":
            self._set_dist_wf(self._config)
        elif self.wf_type == "finetune":
            self._set_ft_wf(self._config)
        elif self.wf_type == "data_gen":
            self._set_data_gen_wf(self._config)

    @property
    def wf_type(self):
        return self._wf_type

    @property
    def download_path(self):
        if isinstance(self._download_path, str):
            return Path(self._download_path)
        else:
            return self._download_path

    def _set_data_gen_wf(self, config):
        default_config = config["default_step_config"]
        prep_fp_config = config["step_configs"].get("perp_fp_config", default_config)
        run_fp_config = config["step_configs"].get("run_fp_config", default_config)
        run_collect_data_config = config["step_configs"].get(
            "collect_data_config", default_config
        )

        upload_python_packages = []
        if custom_packages := config.get("upload_python_packages"):
            upload_python_packages.extend(custom_packages)
        upload_python_packages.extend(list(dpdata.__path__))
        upload_python_packages.extend(list(dflow.__path__))
        upload_python_packages.extend(list(pfd.__path__))
        upload_python_packages.extend(list(dpgen2.__path__))
        upload_python_packages.extend(list(fpop.__path__))

        pert_config = config["conf_generation"]
        if config["conf_generation"]["init_confs"]["confs_uri"] is not None:
            init_confs = get_artifact_from_uri(
                config["conf_generation"]["init_confs"]["confs_uri"]
            )
        elif config["conf_generation"]["init_confs"]["confs_paths"] is not None:
            init_confs_prefix = config["conf_generation"]["init_confs"]["prefix"]
            init_confs = config["conf_generation"]["init_confs"]["confs_paths"]
            init_confs = get_systems_from_data(init_confs, init_confs_prefix)
            init_confs = upload_artifact_and_print_uri(init_confs, "init_confs")
        else:
            raise RuntimeError("init_confs must be provided")

        fp_config = {}
        fp_inputs_config = config["fp"]["inputs_config"]
        fp_type = config["fp"]["type"]
        fp_inputs = fp_styles[fp_type]["inputs"](**fp_inputs_config)
        fp_config["inputs"] = fp_inputs
        fp_config["run"] = config["fp"]["run_config"]
        fp_config["extra_output_files"] = config["fp"].get("extra_output_files")

        data_gen_op = make_data_gen_op(
            fp_style=fp_type,
            prep_fp_config=prep_fp_config,
            run_fp_config=run_fp_config,
            collect_data_config=run_collect_data_config,
            upload_python_packages=upload_python_packages,
        )
        # type_map = config["inputs"]["type_map"]
        data_gen_step = Step(
            "data-gen",
            template=data_gen_op,
            parameters={"pert_config": pert_config, "fp_config": fp_config},
            artifacts={
                "init_confs": init_confs,
            },
        )
        self.workflow.add(data_gen_step)

    def _set_dist_wf(self, config):
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
        collect_data_step_config = config["step_configs"].get(
            "collect_data_config", default_step_config
        )
        pert_gen_step_config = config["step_configs"].get(
            "pert_gen_config", default_step_config
        )
        inference_step_config = config["step_configs"].get(
            "inference_config", copy.deepcopy(run_train_step_config)
        )
        model_test_config = copy.deepcopy(inference_step_config)
        # uploaded python packages
        upload_python_packages = []
        if custom_packages := config.get("upload_python_packages"):
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
        explore_style = config["exploration"]["type"]
        explore_config = config["exploration"]["config"]
        expl_stages = config["exploration"]["stages"]
        converge_config = config["exploration"]["convergence"]
        max_iter = config["exploration"]["max_numb_iter"]
        if max_iter < 1:
            raise RuntimeError(
                "The max number of iteration must be equal to or larger than 1!"
            )
        conf_filters_conv = get_conf_filters_conv(converge_config.pop("conf_filter"))

        # train (student model) style
        train_style = config["train"]["type"]
        train_config = config["train"]["config"]
        # others
        collect_data_config = config["exploration"].get("test_set_config", {})
        collect_data_config["labeled_data"] = collect_data_config.get(
            "labeled_data", False
        )
        collect_data_config["test_size"] = collect_data_config.get("test_size", 0.1)

        ## prepare artifacts
        # read training template
        if isinstance(config["train"]["template_script"], str):
            with open(config["train"]["template_script"], "r") as fp:
                template_script = json.load(fp)
        elif isinstance(config["train"]["template_script"], dict):
            template_script = config["train"]["template_script"]
        else:
            template_script = {}

        # init_confs
        if config["conf_generation"]["init_confs"]["confs_uri"] is not None:
            init_confs = get_artifact_from_uri(
                config["conf_generation"]["init_confs"]["confs_uri"]
            )
        elif config["conf_generation"]["init_confs"]["confs_paths"] is not None:
            init_confs_prefix = config["conf_generation"]["init_confs"]["prefix"]
            init_confs = config["conf_generation"]["init_confs"]["confs_paths"]
            init_confs = get_systems_from_data(init_confs, init_confs_prefix)
            init_confs = upload_artifact_and_print_uri(init_confs, "init_confs")
        else:
            raise RuntimeError("init_confs must be provided")

        # teacher models
        teacher_model_style = config["inputs"]["base_model_style"]
        print("teacher model style: %s" % teacher_model_style)
        teacher_models_paths = config["inputs"]["base_model_path"]
        if config["inputs"]["base_model_uri"] is not None:
            print("Using uploaded model at: ", config["inputs"]["base_model_uri"])
            teacher_models = get_artifact_from_uri(config["inputs"]["base_model_uri"])
        elif teacher_models_paths is not None:
            teacher_models = upload_artifact_and_print_uri(
                teacher_models_paths, "teacher_models"
            )
        else:
            raise FileNotFoundError("Teacher model must exist!")

        inference_config = config["inference"]
        inference_config.update({"model": teacher_model_style})
        inference_config = normalize_infer_args(inference_config)

        expl_args = explore_styles[teacher_model_style][explore_style]["task_args"]
        for stg in expl_stages:
            for task_grp in stg:
                args = expl_args(task_grp)
                task_grp.clear()
                task_grp.update(args)

        scheduler = Scheduler(
            model_style=teacher_model_style,
            explore_style=explore_style,
            explore_stages=expl_stages,
            mass_map=mass_map,
            type_map=type_map,
            max_iter=max_iter,
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
        print("student model style: %s" % train_style)
        # make distillation op
        dist_op = make_dist_op(
            teacher_model_style=teacher_model_style,
            model_style=train_style,
            explore_style=explore_style,
            prep_lmp_config=prep_lmp_step_config,
            run_lmp_config=run_lmp_step_config,
            prep_train_config=prep_train_step_config,
            run_train_config=run_train_step_config,
            collect_data_config=collect_data_step_config,
            pert_gen_config=pert_gen_step_config,
            inference_config=inference_step_config,
            model_test_config=model_test_config,
            upload_python_packages=upload_python_packages,
        )

        # make distillation steps
        dist_step = Step(
            "distillation",
            template=dist_op,
            parameters={
                "block_id": "dist",
                "type_map": type_map,
                "mass_map": mass_map,
                "pert_config": pert_config,
                "numb_models": 1,
                "explore_config": explore_config,
                "converge_config": converge_config,
                "conf_filters_conv": conf_filters_conv,
                "template_script": template_script,
                "train_config": train_config,
                "inference_config": inference_config,
                "test_size": collect_data_config["test_size"],
                "type_map_train": [],
                "scheduler": scheduler,
            },
            artifacts={
                "init_confs": init_confs,
                "teacher_model": teacher_models,
                "init_data": init_data,
                "iter_data": upload_artifact([]),
            },
        )
        self.workflow.add(dist_step)

    def _set_ft_wf(self, config: Dict):
        """
        Build a workflow from the OP templates of model finetune
        """
        default_config = config["default_step_config"]
        prep_train_config = config["step_configs"].get(
            "prep_train_config", default_config
        )
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
        run_pert_gen_config = config["step_configs"].get(
            "pert_gen_config", default_config
        )
        run_inference_config = copy.deepcopy(run_train_config)
        prep_lmp_config = config["step_configs"].get(
            "prep_explore_config", default_config
        )
        run_lmp_config = config["step_configs"].get(
            "run_explore_config", default_config
        )
        # uploaded python packages
        upload_python_packages = []
        if custom_packages := config.get("upload_python_packages"):
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
        train_style = config["train"]["type"]
        train_config = config["train"]["config"]
        if train_style == "dp":
            train_config.update({"init_model_with_finetune": True})
        numb_models = 1
        pert_config = config["conf_generation"]
        explore_style = config["exploration"]["type"]
        expl_stages = config["exploration"]["stages"]
        explore_config = config["exploration"]["config"]
        max_iter = config["exploration"]["max_numb_iter"]
        converge_config = config["exploration"]["convergence"]
        # conf selectors
        conf_filters = get_conf_filters(config["exploration"]["filter"])
        render = TrajRenderLammps(nopbc=False)
        conf_selector = ConfSelectorFrames(
            render, config["fp"]["task_max"], conf_filters
        )
        conf_filters_conv = get_conf_filters_conv(converge_config.pop("conf_filter"))
        # task
        init_training = config["task"]["init_training"]
        if init_training is False:
            print("No initial training before exploration")
        skip_aimd = config["task"]["skip_aimd"]
        if skip_aimd is True:
            print("AIMD exploration is skipped!")
        collect_data_config = {}
        collect_data_config["test_size"] = config["conf_generation"].get(
            "test_data", 0.05
        )
        collect_data_config["system_partition"] = config["conf_generation"].get(
            "system_partition", False
        )
        collect_data_config["labeled_data"] = True
        ## prepare artifacts
        # read training template
        with open(config["train"]["template_script"], "r") as fp:
            template_script = json.load(fp)

        # init_confs
        if config["conf_generation"]["init_confs"]["confs_uri"] is not None:
            init_confs = get_artifact_from_uri(
                config["conf_generation"]["init_confs"]["confs_uri"]
            )
        elif config["conf_generation"]["init_confs"]["confs_paths"] is not None:
            init_confs_prefix = config["conf_generation"]["init_confs"]["prefix"]
            init_confs = config["conf_generation"]["init_confs"]["confs_paths"]
            init_confs = get_systems_from_data(init_confs, init_confs_prefix)
            init_confs = upload_artifact_and_print_uri(init_confs, "init_confs")
        else:
            raise RuntimeError("init_confs must be provided")

        expl_args = explore_styles[train_style][explore_style]["task_args"]
        for stg in expl_stages:
            for task_grp in stg:
                args = expl_args(task_grp)
                task_grp.clear()
                task_grp.update(args)

        scheduler = Scheduler(
            model_style=train_style,
            explore_style=explore_style,
            explore_stages=expl_stages,
            mass_map=mass_map,
            type_map=type_map,
            max_iter=max_iter,
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
        init_models_paths = config["inputs"]["base_model_path"]
        if config["inputs"]["base_model_uri"] is not None:
            print("Using uploaded model at: ", config["inputs"]["base_model_uri"])
            init_models = get_artifact_from_uri(config["inputs"]["base_model_uri"])
        elif init_models_paths:
            init_models = upload_artifact_and_print_uri(init_models_paths, "base_model")
        else:
            raise FileNotFoundError("Pre-trained model must exist!")

        fp_config = {}
        fp_inputs_config = config["fp"]["inputs_config"]
        fp_type = config["fp"]["type"]
        fp_inputs = fp_styles[fp_type]["inputs"](**fp_inputs_config)
        fp_config["inputs"] = fp_inputs
        fp_config["run"] = config["fp"]["run_config"]
        fp_config["extra_output_files"] = config["fp"].get("extra_output_files", [])

        # aimd exploration
        aimd_config = {}
        if skip_aimd is not True:
            aimd_config.update(fp_config)
            aimd_inputs_config = copy.deepcopy(fp_inputs_config)
            aimd_inputs_config.update(config["aimd"]["inputs_config"])
            aimd_inputs = fp_styles[fp_type]["inputs"](**aimd_inputs_config)
            aimd_config["inputs"] = aimd_inputs

        aimd_sample_conf = {"labeled_data": False, "multi_sys_name": "init"}
        if config["aimd"]["confs"]:
            aimd_sample_conf.update(
                {
                    "sample_conf": {
                        "confs": config["aimd"]["confs"],
                        "n_sample": config["aimd"]["n_sample"],
                    }
                }
            )

        inference_config = config["inference"]
        inference_config.update({"model": train_style})
        inference_config = normalize_infer_args(inference_config)
        # get rid of irrelevant argument
        inference_config.pop("max_force")

        # make distillation op
        ft_op = make_ft_op(
            fp_style=fp_type,
            train_style=train_style,
            explore_style=explore_style,
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
            "finetune",
            template=ft_op,
            parameters={
                "block_id": "finetune",
                "type_map": type_map,
                "mass_map": mass_map,
                "pert_config": pert_config,  # Total input parameter file: to be changed in the future
                "numb_models": numb_models,
                "conf_selector": conf_selector,
                "conf_filters_conv": conf_filters_conv,
                "converge_config": converge_config,
                "scheduler": scheduler,
                "explore_config": explore_config,
                "template_script": template_script,
                "train_config": train_config,
                "fp_config": fp_config,
                "aimd_config": aimd_config,
                "aimd_sample_conf": aimd_sample_conf,
                "collect_data_config": collect_data_config,
                "inference_config": inference_config,
            },
            artifacts={
                "init_models": init_models,
                "init_confs": init_confs,
                "expl_models": init_models,
                "init_data": init_data,
                "iter_data": iter_data,
            },
        )
        self.workflow.add(ft_step)

    def _moniter_dist(self):
        print("Running...")
        while True:
            time.sleep(4)
            step_info = self.workflow.query()
            wf_status = self.workflow.query_status()
            if wf_status == "Failed":
                raise RuntimeError(
                    f"Workflow failed (ID: {self.workflow.id}, UID: {self.workflow.uid})"
                )
            try:
                dist_post = step_info.get_step(name="distillation")[0]
            except IndexError:
                continue
            if dist_post["phase"] == "Succeeded":
                print(
                    f"Distillation finished (ID: {self.workflow.id}, UID: {self.workflow.uid})"
                )
                print("Retrieving completed tasks to local...")
                download_artifact(
                    artifact=dist_post.outputs.artifacts["dist_model"],
                    path=self.download_path,
                )
                break

    def _moniter_ft(self):
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
                    artifact=dist_post.outputs.artifacts["fine_tuned_model"],
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
            # return self.workflow
        else:
            return self.workflow
        if not only_submit:
            if self.wf_type == "dist":
                self._moniter_dist()
            elif self.wf_type == "finetune":
                self._moniter_ft()


def successful_step_keys(wf, unsuccessful_step_keys: bool = False):
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


def get_resubmit_keys(wf, unsuccessful_step_keys: bool = False):
    all_step_keys = successful_step_keys(wf, unsuccessful_step_keys)

    step_keys = [
        "pert-gen",
        "sample-aimd",
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
        "inference-test",
        "inference-train",
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
