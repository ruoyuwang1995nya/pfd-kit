import logging
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from dflow import (
    Workflow,
)

from .args import normalize as normalize_args
from .common import (
    global_config_workflow,
)

from pfd.utils.download_pfd_artifacts import (
    download_dpgen2_artifacts,
    download_dpgen2_artifacts_by_def,
)


def download_by_def(
    workflow_id,
    wf_config: Dict = {},
    iterations: Optional[List[int]] = None,
    step_defs: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    chk_pnt: bool = False,
):
    wf_config = normalize_args(wf_config, wf_config["task"]["type"])
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
    wf_config = normalize_args(wf_config, wf_config["task"]["type"])
    global_config_workflow(wf_config)
    wf = Workflow(id=workflow_id)
    if wf_keys is None:
        wf_keys = wf.query_keys_of_steps()
    assert wf_keys is not None
    for kk in wf_keys:
        download_dpgen2_artifacts(wf, kk, prefix=prefix, chk_pnt=chk_pnt)
        logging.info(f"step {kk} downloaded")
