import textwrap
from typing import (
    List,
)

import dargs
from dargs import (
    Argument,
    Variant,
)

from pfd.exploration.converge import CheckConv

from dpgen2.fp import (
    fp_styles,
)
from pfd.train import train_styles

from dpgen2.op.run_lmp import (
    RunLmp,
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


def task_finetune():
    doc_init_train = "Training before exploration"
    doc_skip_aimd = "Skip aimd exploration"
    return [
        Argument(
            "init_training", bool, optional=True, default=False, doc=doc_init_train
        ),
        Argument("skip_aimd", bool, optional=True, default=True, doc=doc_skip_aimd),
    ]


def variant_task():
    return Variant(
        "type",
        [
            Argument("finetune", dict, task_finetune(), alias=["ft"]),
            Argument("dist", dict, [], alias=["distillation"]),
        ],
    )


def conf_args():
    doc_fmt = "Format of input structure files"
    return [
        # Argument("type", str, optional=True, default="file"),
        Argument("prefix", str, optional=True, default=None),
        Argument("fmt", str, optional=True, default="vasp/poscar", doc=doc_fmt),
        Argument("confs_paths", [str, List[str]], optional=True, alias=["files"]),
        Argument("confs_uri", [str, List[str]], optional=True, default=None),
    ]


def pert_gen():
    doc_atom_pert_distance = "Perturb distance for atoms, in Angstrom"
    doc_pert_num = "Number of perturbed structures"
    return [
        Argument("conf_idx", [str, List[int]], optional=True, default="default"),
        Argument(
            "atom_pert_distance",
            float,
            optional=True,
            default=0.1,
            doc=doc_atom_pert_distance,
        ),
        Argument("atom_pert_fraction", float, optional=True, default=0.03),
        Argument("pert_num", int, optional=True, default=1, doc=doc_pert_num),
    ]


def conf_gen_args():
    doc_init_conf = "The initial configurations for PFD workflow"
    doc_pert_gen = "Generation of perturbed structures for PFD exploration"
    return [
        Argument(
            "init_confs",
            dict,
            conf_args(),
            alias=["confs", "init_configurations"],
            doc=doc_init_conf,
        ),
        Argument("pert_generation", [dict, List[dict]], pert_gen(), doc=doc_pert_gen),
    ]


def train_args(run_train):
    doc_numb_models = "Number of models trained for evaluating the model deviation"
    doc_config = "Configuration of training"
    doc_template_script = "File names of the template training script. It can be a `List[str]`, the length of which is the same as `numb_models`. Each template script in the list is used to train a model. Can be a `str`, the models share the same template training script. "
    doc_init_models_paths = "the paths to initial models"
    doc_init_models_uri = "The URI of initial models"
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
        Argument("numb_models", int, optional=True, default=1, doc=doc_numb_models),
        Argument(
            "template_script",
            [List[str], str, dict],
            optional=False,
            doc=doc_template_script,
        ),
        Argument(
            "init_models_paths",
            List[str],
            optional=True,
            default=None,
            doc=doc_init_models_paths,
            alias=["training_iter0_model_path"],
        ),
        Argument(
            "init_models_uri",
            str,
            optional=True,
            default=None,
            doc=doc_init_models_uri,
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
        train_list.append(Argument(kk, dict, train_args(train_styles[kk]["run"])))

    return Variant(
        "type",
        train_list,
        doc=doc,
    )


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


def lmp_args():
    doc_config = "Configuration of lmp exploration"
    doc_max_numb_iter = "Maximum number of iterations per stage"
    doc_fatal_at_max = (
        "Fatal when the number of iteration per stage reaches the `max_numb_iter`"
    )
    doc_output_nopbc = "Remove pbc of the output configurations"
    doc_convergence = "The method of convergence check."
    doc_stages = (
        "The definition of exploration stages of type `List[List[ExplorationTaskGroup]`. "
        "The outer list provides the enumeration of the exploration stages. "
        "Then each stage is defined by a list of exploration task groups. "
        "Each task group is described in :ref:`the task group definition<task_group_sec>` "
    )
    doc_filter = "Filter configuration for DFT calculation"
    doc_conf_filter = (
        "Filtering configurations with too larger or too small prediction error"
    )

    return [
        Argument(
            "config",
            dict,
            RunLmp.lmp_args(),
            optional=True,
            default=RunLmp.normalize_config({}),
            doc=doc_config,
        ),
        Argument(
            "max_numb_iter",
            int,
            optional=True,
            default=10,
            doc=doc_max_numb_iter,
            alias=["max_iter"],
        ),
        Argument(
            "fatal_at_max", bool, optional=True, default=True, doc=doc_fatal_at_max
        ),
        Argument(
            "output_nopbc", bool, optional=True, default=False, doc=doc_output_nopbc
        ),
        Argument(
            "convergence",
            dict,
            [
                Argument(
                    "conf_filter",
                    List[dict],
                    optional=True,
                    default=[],
                    doc=doc_conf_filter,
                )
            ],
            [variant_conv()],
            optional=False,
            doc=doc_convergence,
            alias=["converge_config"],
        ),
        Argument(
            "filter",
            List[dict],
            optional=True,
            default=[{"type": "distance"}],
            doc=doc_filter,
        ),
        Argument("stages", List[List[dict]], optional=False, doc=doc_stages),
    ]


def variant_explore():
    doc = "The type of the exploration"
    doc_lmp = "The exploration by LAMMPS simulations"
    return Variant(
        "type",
        [
            Argument("lmp", dict, lmp_args(), doc=doc_lmp),
        ],
        doc=doc,
    )


def fp_args(inputs, run):
    doc_inputs_config = "Configuration for preparing vasp inputs"
    doc_run_config = "Configuration for running vasp tasks"
    doc_task_max = "Maximum number of vasp tasks for each iteration"

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
        Argument("task_max", int, optional=True, default=100, doc=doc_task_max),
        Argument("extra_output_files", List, optional=True, default=[]),
    ]


def variant_fp():
    doc = "the type of the fp"

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


def input_args():
    """
    The input parameters and artifacts of PFD workflow
    """
    doc_type_map = 'The type map. e.g. ["Al", "Mg"]. Element in dpdata format.'
    doc_mass_map = "The mass map. e.g. [27., 24.]. Al and Mg will be set with mass 27. and 24. amu, respectively."
    # doc_mixed_type = "Use `deepmd/npy/mixed` format for storing training data."
    doc_init_data_prefix = "The prefix of initial data systems"
    doc_init_sys = "The inital data systems"
    doc_init_data_uri = "The URI of initial data"
    # Hide these settings in the input section
    # doc_multitask = "Do multitask training"
    # doc_head = "Head to use in the multitask training"
    # doc_multi_init_data = (
    #    "The inital data for multitask, it should be a dict, whose keys are task names and each value is a dict"
    #    "containing fields `prefix` and `sys` for initial data of each task"
    # )
    # doc_multi_init_data_uri = "The URI of initial data for multitask"
    doc_valid_data_prefix = "The prefix of validation data systems"
    doc_valid_sys = "The validation data systems"
    doc_valid_data_uri = "The URI of validation data"
    doc_base_model_paths = (
        "Path to the base model."
        "In `finetune` task, this is the path to the pretrained model."
        "In `distillation` task, this is the path to the teacher model."
    )
    doc_base_model_paths = textwrap.dedent(doc_base_model_paths)
    doc_base_model_uri = "URI of the base model."
    doc_base_model_style = "Type of the base model"
    # doc_student_model_style = "Type of the student model in `distillation` task"

    return [
        Argument("type_map", List[str], optional=False, doc=doc_type_map),
        Argument("mass_map", List[float], optional=True, doc=doc_mass_map),
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
            "valid_data_prefix",
            str,
            optional=True,
            default=None,
            doc=doc_valid_data_prefix,
        ),
        Argument(
            "valid_data_sys",
            [List[str], str],
            optional=True,
            default=None,
            doc=doc_valid_sys,
        ),
        Argument(
            "valid_data_uri",
            str,
            optional=True,
            default=None,
            doc=doc_valid_data_uri,
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
        Argument(
            "base_model_style",
            str,
            optional=True,
            default="dp",
            doc=doc_base_model_style,
            alias=["teacher_model_style"],
        ),
    ]


def aimd_args():
    doc_conf = "The systems selected for initial fp calculation"
    doc_n_sample = (
        "The number of configurations selected for fp calculation within each system"
    )
    return [
        Argument("confs", List[int], optional=True, default=None, doc=doc_conf),
        Argument("n_sample", int, optional=True, default=1, doc=doc_n_sample),
    ]


def infer_args():
    doc_max_force = "The max value of allowed atomic force"
    return [
        Argument("max_force", float, optional=True, default=None, doc=doc_max_force)
    ]


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

    return [
        Argument(
            "prep_train_config",
            dict,
            step_conf_args(),
            optional=True,
            default=default_config,
            doc=doc_prep_train_config,
        ),
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


def task_args():
    doc_task = "Task type, `finetune` or `dist`"
    doc_inputs = "The input parameter and artifacts for pfd"
    return [
        Argument("task", dict, [], [variant_task()], optional=False, doc=doc_task),
        Argument("inputs", dict, input_args(), optional=False, doc=doc_inputs),
    ]


def conf_generation_args():
    doc_conf_gen = "The inputparameter and artifacts for confs generation"
    return [
        Argument(
            "conf_generation",
            dict,
            conf_gen_args(),
            # pert_gen(),
            optional=False,
            alias=["configurations"],
            doc=doc_conf_gen,
        ),
    ]


def training_args():
    doc_train = "The configuration for training"
    return [
        Argument("train", dict, [], [variant_train()], optional=False, doc=doc_train),
    ]


def label_args():
    doc_fp = "The configuration for FP"
    doc_aimd = "The parameter for initial fp calculation"
    doc_infer = "The parameters for inference settings"
    return [
        Argument("fp", dict, [], [variant_fp()], optional=True, doc=doc_fp),
        Argument("aimd", dict, aimd_args(), optional=True, doc=doc_aimd),
        Argument(
            "inference",
            dict,
            infer_args(),
            optional=True,
            default={},
            doc=doc_infer,
        ),
    ]


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


def submit_args(default_step_config=normalize_step_dict({})):
    return (
        wf_args(default_step_config)
        + task_args()
        + conf_generation_args()
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
