from calendar import c
import dflow
from dflow import Workflow, Step,upload_artifact
from pfd.exploration.task.stage import ExplorationStage
from pfd.op import (
    CollectData,
    SelectConfs,
    ModelTestOP,
    PrepASE,
    RunASE,
    TrainFoo,
    StageSchedulerDist,
    StageSchedulerFT
    )
from pfd.flow.expl_train import ExplTrainBlock, ExplTrainLoop
from pfd.superop import (
    PrepRunExpl,
    PrepRunFp
)
from pfd.fp import (
    PrepFoo,
    RunFoo
)
from pfd.exploration.task import AseTaskGroup
from pfd.exploration.render import TrajRender
from pfd.exploration.selector import ConfSelectorFrames
from pfd.exploration.scheduler import Scheduler
from pfd.entrypoint.submit import FlowGen

from pathlib import Path

from ase.build import bulk
from ase.io import write,read

from io import StringIO
# Set up the workflow and steps

#dflow.config["mode"] = "debug"
# if submit to kubernetes
# form dflow.config import config, s3_config
# s3_config['storage_client'] set the storage for artifacts

import json



if __name__ == "__main__":
    with open('./flow_test_ft.json','r') as f:
        config = json.load(f)
        
    print(config["task"])
    wf=FlowGen(
        config=config,
        debug=True
    )
    print(dflow.config)
    wf.submit()
