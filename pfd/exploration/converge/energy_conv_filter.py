from .conf_filter_conv import ConfFilterConv
from pfd.exploration.inference import TestReport
import dpdata
import numpy as np
from typing import Dict


@ConfFilterConv.register("energy_delta")
class EnerConfFilter(ConfFilterConv):
    def __init__(self, thr_l: float = 0.0, thr_h: float = 1) -> None:
        self.thr_l = thr_l
        self.thr_h = thr_h

    def check(self, rep: TestReport):
        if rep.mae_e > self.thr_h or rep.mae_e < self.thr_l:
            return False
        return True


@ConfFilterConv.register("force_delta")
class ForceConfFilter(ConfFilterConv):
    def __init__(self, thr_l: float = 0.0, thr_h: float = 0.3) -> None:
        self.thr_l = thr_l
        self.thr_h = thr_h

    def check(self, rep: TestReport):
        if rep.mae_f > self.thr_h or rep.mae_f < self.thr_l:
            return False
        return True
