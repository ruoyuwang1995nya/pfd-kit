import textwrap
from typing import List, Union

import dargs
from dargs import (
    Argument,
    Variant,
)

# from pfd.exploration.converge import CheckConv, ConfFilterConv
from pfd.exploration.selector import conf_filter_styles
from pfd.exploration.converge import CheckConv
from pfd.exploration.inference import EvalModel
from pfd.fp import (
    fp_styles,
)
from pfd.train import train_styles

from pfd.op.run_md import (
    RunASE,
)
from pfd.utils import (
    normalize_step_dict,
    step_conf_args,
)


def make_link(content, ref_key):
    raw_anchor = dargs.dargs.RAW_ANCHOR
    return (
        f"`{content} <{ref_key}_>`_" if not raw_anchor else f"`{content} <#{ref_key}>`_"
    )


def conf_args():
    doc_fmt = "ASE compatible format of input structure files"
    return [
        Argument("prefix", str, optional=True, default=None),
        Argument("fmt", str, optional=True, default="extxyz", doc=doc_fmt),
        Argument(
            "confs_paths",
            [str, List[str]],
            optional=True,
            default=None,
            alias=["files"],
        ),
        Argument("confs_uri", [str, List[str]], optional=True, default=None),
    ]


#### task config
def task_args():
    doc_task = "Task type, `finetune` or `dist`"
    doc_max_iter = "Maximum number of iterations"
    doc_init_fp = "Initialize fine-tuning"
    doc_init_train = "Initialize training"
    return [
        Argument("type", str, optional=False, doc=doc_task),
        Argument("max_iter", int, optional=True, default=1, doc=doc_max_iter),
        Argument("init_fp", bool, optional=True, default=False, doc=doc_init_fp),
        Argument("init_train", bool, optional=True, default=False, doc=doc_init_train),
    ]


#### inputs config
def inputs_args():
    """
    The input parameters and artifacts of PFD workflow
    """
    doc_init_data_prefix = "The prefix of initial data systems"
    doc_init_sys = "The inital data systems"
    doc_init_data_uri = "The URI of initial data"
    doc_base_model_paths = (
        "Path to the base model."
        "In `finetune` task, this is the path to the pretrained model."
        "In `distillation` task, this is the path to the teacher model."
    )
    doc_base_model_paths = textwrap.dedent(doc_base_model_paths)
    doc_base_model_uri = "URI of the base model."
    doc_init_confs = "The initial configurations for exploration"
    doc_init_fp_confs = "The configurations for initial first-principles calculations"
    return [
        Argument("init_confs", dict, conf_args(), optional=False, doc=doc_init_confs),
        Argument(
            "init_fp_confs",
            dict,
            conf_args(),
            optional=True,
            default={},
            doc=doc_init_fp_confs,
        ),
        Argument(
            "init_data_prefix",
            str,
            optional=True,
            default=None,
            doc=doc_init_data_prefix,
        ),
        # Argument("mixed_type", bool, optional=True, default=False, doc=doc_mixed_type),
        Argument(
            "init_data_sys",
            [List[str], str],
            optional=True,
            default=None,
            doc=doc_init_sys,
        ),
        Argument(
            "init_data_uri",
            str,
            optional=True,
            default=None,
            doc=doc_init_data_uri,
        ),
        Argument(
            "base_model_path",
            [List[str], str],
            optional=True,
            default=None,
            alias=["teacher_model_path", "pretrain_model_path", "teacher_models_paths"],
            doc=doc_base_model_paths,
        ),
        Argument(
            "base_model_uri",
            str,
            optional=True,
            default=None,
            alias=["teacher_model_uri", "pretrain_model_uri"],
            doc=doc_base_model_uri,
        ),
    ]


#### Explore
def ase_args():
    doc_stages = (
        "Exploration stages."
        "The definition of exploration stages of type `List[List[ExplorationTaskGroup]`. "
        "The outer list provides the enumeration of the exploration stages. "
        "Then each stage is defined by a list of exploration task groups. "
        "Each task group is described in :ref:`the task group definition<task_group_sec>` "
    )
    doc_config = "Configuration of ase exploration"
    return [
        Argument(
            "config", dict, RunASE.ase_args(), optional=True, default={}, doc=doc_config
        ),
        Argument("stages", List[List[dict]], optional=False, doc=doc_stages),
    ]


def caly_args():
    doc_stages = (
        "Exploration stages."
        "The definition of exploration stages of type `List[List[ExplorationTaskGroup]`. "
        "The outer list provides the enumeration of the exploration stages. "
        "Then each stage is defined by a list of exploration task groups. "
        "Each task group is described in :ref:`the task group definition<task_group_sec>` "
    )
    doc_config = "Configuration of ase exploration"
    doc_run_calypso_command = "command of running calypso."
    return [
        Argument(
            "config",
            dict,
            RunASE.ase_args()
            + [
                Argument(
                    "run_calypso_command",
                    str,
                    optional=True,
                    default="calypso.x",
                    doc=doc_run_calypso_command,
                ),
            ],
            doc=doc_config,
        ),
        Argument("stages", List[List[dict]], optional=False, doc=doc_stages),
    ]


def variant_explore():
    doc = "The type of the exploration"
    doc_ase = "Exploration by ASE"
    doc_calypso = "Exploration by Calypso"
    return Variant(
        "type",
        [
            Argument("ase", dict, ase_args(), doc=doc_ase),
            Argument("calypso", dict, caly_args(), doc=doc_calypso),
            Argument("calypso:merge", dict, caly_args(), doc=doc_calypso),
        ],
        doc=doc,
    )


def explore_args():
    doc_test_set = "Set the portion of test set. Only available for `dist`"
    doc_explore = "The configuration for exploration"
    return [
        Argument(
            "exploration",
            dict,
            [
                Argument(
                    "test_set_config",
                    dict,
                    optional=True,
                    default={"test_size": 0.1},
                    alias=["test_set"],
                    doc=doc_test_set,
                )
            ],
            [variant_explore()],
            optional=False,
            doc=doc_explore,
            alias=["explore"],
        ),
    ]


#### FP calculation
def fp_args(inputs, run):
    doc_inputs_config = "Configuration for preparing vasp inputs"
    doc_run_config = "Configuration for running vasp tasks"
    doc_extra_output_files = "Extra output file names, support wildcards"

    return [
        Argument(
            "inputs_config",
            dict,
            inputs.args(),
            optional=False,
            doc=doc_inputs_config,
        ),
        Argument(
            "run_config",
            dict,
            run.args(),
            optional=False,
            doc=doc_run_config,
        ),
        Argument(
            "extra_output_files",
            List,
            optional=True,
            default=[],
            doc=doc_extra_output_files,
        ),
    ]


def variant_fp():
    doc = "Tpyes of first-principles calculators"
    fp_list = []
    for kk in fp_styles.keys():
        fp_list.append(
            Argument(
                kk,
                dict,
                fp_args(fp_styles[kk]["inputs"], fp_styles[kk]["run"]),
            )
        )

    return Variant("type", fp_list, doc=doc)


def label_args():
    doc_fp = "The configuration for FP"
    return [
        Argument("fp", dict, [], [variant_fp()], optional=True, doc=doc_fp),
    ]


#### train config
def train_args(run_train):
    """[Modified from DPGEN2] General train config"""
    doc_config = "Configuration of training"
    doc_template_script = "File names of the template training script. It can be a `List[str]`, the length of which is the same as `numb_models`. Each template script in the list is used to train a model. Can be a `str`, the models share the same template training script. "
    doc_optional_files = "Optional files for training"

    return [
        Argument(
            "config",
            dict,
            run_train.training_args(),
            optional=True,
            default=run_train.normalize_config({}),
            doc=doc_config,
        ),
        Argument(
            "template_script",
            [List[str], str, dict],
            optional=True,
            default={},
            doc=doc_template_script,
        ),
        Argument(
            "optional_files",
            list,
            optional=True,
            default=None,
            doc=doc_optional_files,
        ),
    ]


def variant_train():
    doc = "the type of the training model"
    train_list = []
    for kk in train_styles.keys():
        train_list.append(Argument(kk, dict, train_args(train_styles[kk])))
    return Variant(
        "type",
        train_list,
        doc=doc,
    )


def training_args():
    doc_train = "The configuration for training"
    return [
        Argument("train", dict, [], [variant_train()], optional=False, doc=doc_train),
    ]


#### evaluate config
def variant_conv():
    doc = "the type of the condidate selection and convergence check method."
    var_list = []
    for kk, vv in CheckConv.get_checkers().items():
        var_list.append(Argument(kk, dict, vv.args(), doc=vv.doc()))
    return Variant(
        "type",
        var_list,
        doc=doc,
    )


def evaluate_args():
    doc_max_sel = "Maximum number of selected configurations"
    doc_model = (
        "The model type used in the evaluation. "
        "It should be consistent with the model type used in training."
    )
    doc_converge = "The method of convergence check."
    return [
        Argument("max_sel", int, optional=True, default=50, doc=doc_max_sel),
        Argument("model", str, optional=True, default="dp", doc=doc_model),
        Argument(
            "converge",
            dict,
            [],
            [variant_conv()],
            optional=True,
            default={},
            doc=doc_converge,
        ),
    ]


#### select confs config
def variant_frame_selector():
    doc = "the type of the frame selector"
    var_list = []
    for kk, vv in conf_filter_styles.items():
        var_list.append(Argument(kk, dict, vv.args(), doc=vv.doc()))
    return Variant("type", var_list, doc=doc)


def h_filter_args():
    doc_k = "Number of nearest neighbors to consider"
    doc_cutoff = "Cutoff distance (in unit of angstrom)"
    doc_batch_size = "Batch size for calculating the similarity matrix"
    doc_h = (
        "Bandwidth of the Gaussian kernel (in unit of angstrom)."
        "It controls the level of 'similarity' between two configurations"
    )
    doc_chunksize = "The chunk size of adding new configurations."
    return [
        Argument("k", int, optional=True, default=32, doc=doc_k),
        Argument("cutoff", float, optional=True, default=5.0, doc=doc_cutoff),
        Argument("batch_size", int, optional=True, default=1000, doc=doc_batch_size),
        Argument("h", float, optional=True, default=0.015, doc=doc_h),
        Argument("chunk_size", int, optional=True, default=10, doc=doc_chunksize),
    ]


def select_confs_args():
    doc_test_size = (
        "The number of data frames split from training data as test set."
        "If `test_size<1`, it is the portion of test set. If `test_size>=1`,"
        "it is the number of frames in the test set."
    )
    doc_h_filter = "Select configurations based on entropy contribution"
    return [
        Argument("test_size", float, optional=True, default=0.1, doc=doc_test_size),
        Argument(
            "frame_filter",
            List[dict],
            [],
            [variant_frame_selector()],
            optional=True,
            default=[],
        ),
        Argument(
            "h_filter",
            dict,
            h_filter_args(),
            optional=True,
            default=None,
            doc=doc_h_filter,
        ),
    ]


#### dflow related
def dflow_conf_args():
    doc_dflow_config = "The configuration passed to dflow"
    doc_dflow_s3_config = "The S3 configuration passed to dflow"

    return [
        Argument(
            "dflow_config", dict, optional=True, default=None, doc=doc_dflow_config
        ),
        Argument(
            "dflow_s3_config",
            dict,
            optional=True,
            default=None,
            doc=doc_dflow_s3_config,
        ),
    ]


def bohrium_conf_args():
    doc_username = "The username of the Bohrium platform"
    doc_password = "The password of the Bohrium platform"
    doc_project_id = "The project ID of the Bohrium platform"
    doc_host = (
        "The host name of the Bohrium platform. Will overwrite `dflow_config['host']`"
    )
    doc_k8s_api_server = "The k8s server of the Bohrium platform. Will overwrite `dflow_config['k8s_api_server']`"
    doc_repo_key = "The repo key of the Bohrium platform. Will overwrite `dflow_s3_config['repo_key']`"
    doc_storage_client = "The storage client of the Bohrium platform. Will overwrite `dflow_s3_config['storage_client']`"

    return [
        Argument("username", str, optional=False, doc=doc_username),
        Argument("password", str, optional=True, doc=doc_password),
        Argument("project_id", int, optional=False, doc=doc_project_id),
        Argument("ticket", str, optional=True),
        Argument(
            "host",
            str,
            optional=True,
            default="https://workflows.deepmodeling.com",
            doc=doc_host,
        ),
        Argument(
            "k8s_api_server",
            str,
            optional=True,
            default="https://workflows.deepmodeling.com",
            doc=doc_k8s_api_server,
        ),
        Argument(
            "repo_key", str, optional=True, default="oss-bohrium", doc=doc_repo_key
        ),
        Argument(
            "storage_client",
            str,
            optional=True,
            default="dflow.plugins.bohrium.TiefblueClient",
            doc=doc_storage_client,
        ),
    ]


def default_step_config_args():
    doc_default_step_config = "The default step configuration."

    return [
        Argument(
            "default_step_config",
            dict,
            step_conf_args(),
            optional=True,
            default={},
            doc=doc_default_step_config,
        ),
    ]


def pfd_step_config_args(default_config):
    doc_prep_train_config = "Configuration for prepare train"
    doc_run_train_config = "Configuration for run train"
    doc_prep_explore_config = "Configuration for prepare exploration"
    doc_run_explore_config = "Configuration for run exploration"
    doc_prep_fp_config = "Configuration for prepare fp"
    doc_run_fp_config = "Configuration for run fp"
    doc_select_confs_config = "Configuration for the select confs"
    doc_collect_data_config = "Configuration for the collect data"
    doc_evaluate_config = "Configuration for model evaluation"

    return [
        Argument(
            "run_train_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_run_train_config,
        ),
        Argument(
            "prep_explore_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_prep_explore_config,
        ),
        Argument(
            "run_explore_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_run_explore_config,
        ),
        Argument(
            "prep_fp_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_prep_fp_config,
        ),
        Argument(
            "run_fp_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_run_fp_config,
        ),
        Argument(
            "select_confs_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_select_confs_config,
        ),
        Argument(
            "collect_data_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_collect_data_config,
        ),
        Argument(
            "evaluate_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_evaluate_config,
        ),
    ]


def wf_args(default_step_config=normalize_step_dict({})):
    doc_name = "The workflow name, 'pfd' for default"
    doc_bohrium_config = "Configurations for the Bohrium platform."
    doc_step_configs = "Configurations for executing dflow steps"
    doc_upload_python_packages = "Upload python package, for debug purpose"
    doc_parallelism = "The parallelism for the workflow. Accept an int that stands for the maximum number of running pods for the workflow. None for default"
    return (
        [Argument("name", str, optional=True, default="pfd", doc=doc_name)]
        + dflow_conf_args()
        + default_step_config_args()
        + [
            Argument(
                "parallelism", int, optional=True, default=None, doc=doc_parallelism
            ),
            Argument(
                "bohrium_config",
                dict,
                bohrium_conf_args(),
                optional=True,
                default=None,
                doc=doc_bohrium_config,
            ),
            Argument(
                "step_configs",
                dict,
                pfd_step_config_args(default_step_config),
                optional=True,
                default={},
                doc=doc_step_configs,
            ),
            Argument(
                "upload_python_packages",
                [List[str], str],
                optional=True,
                default=None,
                doc=doc_upload_python_packages,
                alias=["upload_python_package"],
            ),
        ]
    )


def submit_args(default_step_config=normalize_step_dict({})):
    """Normalize the full input arguments of the submit script

    Args:
        default_step_config (_type_, optional): _description_. Defaults to normalize_step_dict({}).

    Returns:
        _type_: _description_
    """
    return (
        wf_args(default_step_config)
        + [
            Argument("task", dict, task_args()),
            Argument("inputs", dict, inputs_args()),
            Argument(
                "select_confs", dict, select_confs_args(), optional=True, default={}
            ),
            Argument("evaluate", dict, evaluate_args(), optional=True, default={}),
        ]
        + training_args()
        + label_args()
        + explore_args()
    )


def normalize(data):
    default_step_config = normalize_step_dict(data.get("default_step_config", {}))
    defs = submit_args(default_step_config)
    base = Argument("base", dict, defs)
    data = base.normalize_value(data, trim_pattern="_*")
    # not possible to strictly check arguments, dirty hack!
    base.check_value(data, strict=False)
    return data


def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    """[Modified from DPGEN2]Generate the doc string of the submit args

    Args:
        make_anchor (bool, optional): _description_. Defaults to True.
        make_link (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if make_link:
        make_anchor = True
    sca = submit_args()
    base = Argument("submit", dict, sca)
    ptr = []
    ptr.append(base.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))
    key_words = []
    for ii in "\n\n".join(ptr).split("\n"):
        if "argument path" in ii:
            key_words.append(ii.split(":")[1].replace("`", "").strip())
    return "\n\n".join(ptr)
