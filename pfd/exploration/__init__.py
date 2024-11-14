from dpgen2.superop import PrepRunLmp
from dpgen2.op import PrepLmp, RunLmp
from .task import gen_expl_grp_lmps, expl_grp_args_lmps

# from .scheduler import Scheduler


explore_styles = {
    "dp": {
        "lmp": {
            "preprun": PrepRunLmp,
            "prep": PrepLmp,
            "run": RunLmp,
            "task": gen_expl_grp_lmps,
            "task_args": expl_grp_args_lmps,
        }
    }
}
