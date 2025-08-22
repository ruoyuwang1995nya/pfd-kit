from dpgen2.superop import PrepRunLmp
from dpgen2.op import PrepLmp, RunLmp

from .task import (
    gen_expl_grp_lmps,
    expl_grp_args_lmps,
    gen_expl_grp_caly,
    expl_grp_args_caly,
)

# from .scheduler import Scheduler


explore_styles = {
    "lmp": {
        "dp": {
            "preprun": PrepRunLmp,
            "prep": PrepLmp,
            "run": RunLmp,
            "task": gen_expl_grp_lmps,
            "task_args": expl_grp_args_lmps,
        },
        "calypso": {"task": gen_expl_grp_caly, "task_args": expl_grp_args_caly},
    },
    "ase":{
        "dp": {
            "preprun": None,
            "prep": None,
            "run": None,
            "task": None,
            "task_args": None,
        },
        "calypso": {"task": None, "task_args": None},
    }
}
