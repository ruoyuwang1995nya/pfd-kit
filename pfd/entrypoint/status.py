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

from pfd.entrypoint.args import normalize as normalize_args
from pfd.entrypoint.common import (
    global_config_workflow,
)
from pfd.utils.dflow_query import (
    get_last_scheduler,
)


def status(
    workflow_id,
    wf_config: Optional[Dict] = {},
):
    wf_config = normalize_args(wf_config)

    global_config_workflow(wf_config)

    wf = Workflow(id=workflow_id)

    wf_keys = wf.query_keys_of_steps()

    scheduler = get_last_scheduler(wf, wf_keys)

    if scheduler is not None:
        ptr_str = scheduler.get_status()
        print(ptr_str)
    else:
        logging.warning("no scheduler is finished")
