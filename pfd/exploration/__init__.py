from pfd.superop import PrepRunExpl
from pfd.op import (
    PrepASE,
    RunASE
)
from .task import (
    AseTaskGroup
)

# from .scheduler import Scheduler


explore_styles = {
    "ase":{
        "preprun": PrepRunExpl,
        "prep": PrepASE,
        "run": RunASE,
        "task_grp": AseTaskGroup,
    }
}
