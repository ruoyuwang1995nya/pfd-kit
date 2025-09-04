from pfd.op.stage import StageSchedulerDist, StageSchedulerFT
from .expl_train import ExplTrainLoop, ExplTrainBlock
from .flow import PFD
from .data_gen import DataGen
wf_styles = {
    "finetune": StageSchedulerFT,
    "dist": StageSchedulerDist
}