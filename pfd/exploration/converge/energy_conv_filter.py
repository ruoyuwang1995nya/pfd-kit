from .conf_filter_conv import ConfFilterConv
from pfd.exploration.inference import TestReport
import logging
from dargs import Argument


@ConfFilterConv.register("energy_delta")
class EnerConfFilter(ConfFilterConv):
    def __init__(self, thr_l: float = 0.0, thr_h: float = 1) -> None:
        self.thr_l = thr_l
        self.thr_h = thr_h

    def check(self, rep: TestReport):
        if rep.mae_e_atom < self.thr_l:
            logging.warning(f"#### Energy predition error ({rep.mae_e_atom}) out of low threshold ({self.thr_l})!")
            return False
        elif rep.mae_e_atom > self.thr_h:
            logging.warning(f"#### Energy predition error ({rep.mae_e_atom}) out of high threshold ({self.thr_h})!")
            return False
        return True

    @classmethod
    def args(cls):
        doc_thr_l = "The lower threshold of the energy/atom prediction error"
        doc_thr_h = "The higher threshold of the energy/atom prediction error"
        return [
            Argument("thr_l", float, optional=True, default=0.0, doc=doc_thr_l),
            Argument("thr_h", float, optional=True, default=1.0, doc=doc_thr_h),
        ]

    @classmethod
    def doc(cls):
        return "allowed prediction error of energy/atom, in eV/atom"


@ConfFilterConv.register("force_delta")
class ForceConfFilter(ConfFilterConv):
    def __init__(self, thr_l: float = 0.0, thr_h: float = 0.3) -> None:
        self.thr_l = thr_l
        self.thr_h = thr_h

    def check(self, rep: TestReport):
        if rep.rmse_f < self.thr_l:
            logging.warning(f"#### Force predition error ({rep.rmse_f}) out of low threshold ({self.thr_l})!")
            return False
        elif rep.rmse_f > self.thr_h:
            logging.warning(f"#### Force predition error ({rep.rmse_f}) out of high threshold ({self.thr_h})!")
            return False
        return True

    @classmethod
    def args(cls):
        doc_thr_l = "The lower threshold of the atomic forces prediction error"
        doc_thr_h = "The higher threshold of the atomic forces prediction error"
        return [
            Argument("thr_l", float, optional=True, default=0.0, doc=doc_thr_l),
            Argument("thr_h", float, optional=True, default=0.3, doc=doc_thr_h),
        ]

    @classmethod
    def doc(cls):
        return "allowed prediction error of average atomic forces, in eV/Angstrom"
