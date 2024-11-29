import sys

sys.path.append("../")
import unittest
import copy
from math import sqrt
from pfd.op.stage import StageScheduler
import dpdata
from pathlib import Path
import numpy as np
from dflow.python import OPIO
from pfd.exploration import explore_styles
from pfd.exploration.scheduler import Scheduler
from pfd.exploration.inference import TestReport, TestReports


class TestTestReport(unittest.TestCase):
    def setUp(self):
        self.rep0 = TestReport(
            name="sys0",
            atom_numb=1,
            numb_frame=1,
            mae_e_atom=0.01,
            mae_f=0.1,
            rmse_e=0.02,
            rmse_f=0.2,
        )

        self.rep1 = TestReport(
            name="sys1",
            atom_numb=2,
            numb_frame=2,
            mae_e_atom=0.02,
            mae_f=0.2,
            rmse_e=0.04,
            rmse_f=0.4,
        )

        self.reports = TestReports()
        self.reports.add_report(self.rep0)
        self.reports.add_report(self.rep1)

    def test_test_report(self):
        rmse_f = sqrt(
            (
                3 * self.rep0.numb_frame * self.rep0.atom_numb * self.rep0.rmse_f**2
                + 3 * self.rep1.numb_frame * self.rep1.atom_numb * self.rep1.rmse_f**2
            )
            / (
                3 * self.rep0.numb_frame * self.rep0.atom_numb
                + 3 * self.rep1.numb_frame * self.rep1.atom_numb
            )
        )
        self.assertAlmostEqual(self.reports.get_weighted_rmse_f(), rmse_f)

        rmse_e = sqrt(
            (
                self.rep0.numb_frame * self.rep0.atom_numb * self.rep0.rmse_e_atom**2
                + self.rep1.numb_frame
                * self.rep1.atom_numb
                * self.rep1.rmse_e_atom**2
            )
            / (
                3 * self.rep0.numb_frame * self.rep0.atom_numb
                + 3 * self.rep1.numb_frame * self.rep1.atom_numb
            )
        )

        self.assertAlmostEqual(self.reports.get_weighted_rmse_e_atom(), rmse_e)


if __name__ == "__main__":
    unittest.main()
