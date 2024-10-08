from dpgen2.superop import PrepRunLmp
from dpgen2.op import PrepLmp, RunLmp
from pfd.exploration.task import gen_expl_grp_lmps

explore_styles = {
    "dp": {
        "lmp": {
            "preprun": PrepRunLmp,
            "prep": PrepLmp,
            "run": RunLmp,
            "task": gen_expl_grp_lmps,
        }
    }
}
