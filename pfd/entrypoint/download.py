import logging
from typing import (
    Dict,
    List,
    Optional,
    Union,
)
from pathlib import Path
from dflow import Workflow, download_artifact

from .args import normalize as normalize_args
from .common import (
    global_config_workflow,
)

from pfd.utils.download_pfd_artifacts import (
    download_dpgen2_artifacts,
    download_dpgen2_artifacts_by_def,
)


END_OUTPUT = {
    "finetune": {"steps": "finetune", "artifact": "fine_tuned_model"},
    "dist": {"steps": "distillation", "artifact": "dist_model"},
}


def download_end_result(
    workflow_id, wf_config: Dict = {}, prefix: Optional[str] = None
):
    try:
        task_type = wf_config["task"]["type"]
    except KeyError:
        raise RuntimeError("Illegal input file!")
    wf_config = normalize_args(wf_config)
    global_config_workflow(wf_config)
    wf = Workflow(id=workflow_id)
    step_info = wf.query()
    wf_status = wf.query_status()
    if wf_status == "Failed":
        raise RuntimeError(f"Workflow failed (ID: {wf.id}, UID: {wf.uid})")
    try:
        wf_post = step_info.get_step(name=END_OUTPUT[task_type]["steps"])[0]
    except IndexError:
        logging.warning("The workflow may not have finished!")
        return
    if wf_post["phase"] == "Succeeded":
        print(f"Workflow finished (ID: {wf.id}, UID: {wf.uid})")
        print("Retrieving completed tasks to local...")
        if prefix is not None:
            path = Path(prefix)
        else:
            path = Path("./results")
        download_artifact(
            artifact=wf_post.outputs.artifacts[END_OUTPUT[task_type]["artifact"]],
            path=path,
        )


def download_by_def(
    workflow_id,
    wf_config: Dict = {},
    iterations: Optional[List[int]] = None,
    step_defs: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    chk_pnt: bool = False,
):
    wf_config = normalize_args(wf_config)
    global_config_workflow(wf_config)
    wf = Workflow(id=workflow_id)
    download_dpgen2_artifacts_by_def(wf, iterations, step_defs, prefix, chk_pnt)


def download(
    workflow_id,
    wf_config: Optional[Dict] = {},
    wf_keys: Optional[List] = None,
    prefix: Optional[str] = None,
    chk_pnt: bool = False,
):
    wf_config = normalize_args(wf_config)
    global_config_workflow(wf_config)
    wf = Workflow(id=workflow_id)
    if wf_keys is None:
        wf_keys = wf.query_keys_of_steps()
    assert wf_keys is not None
    for kk in wf_keys:
        download_dpgen2_artifacts(wf, kk, prefix=prefix, chk_pnt=chk_pnt)
        logging.info(f"step {kk} downloaded")
