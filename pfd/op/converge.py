from genericpath import isdir
from pathlib import Path
from pathlib import (
    Path,
)
from typing import List, Dict

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
from pfd.exploration import converge
from pfd.exploration.converge import CheckConv, ConfFiltersConv, ConvReport
from pfd.exploration.inference import TestReport, TestReports
from pfd.exploration.scheduler import Scheduler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("converge.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class EvalConv(OP):
    """
    Args:
        converged: boolean, whether the workflow has already converged.
        systems: dpdata system, a list of systems
    """

    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "converged": Parameter(bool, default=False),
                "config": Parameter(dict, default={}),
                "systems": Artifact(List[Path], optional=True),
                "test_res": BigParameter(TestReports),
                "conf_filters_conv": BigParameter(ConfFiltersConv, default=None),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "converged": Parameter(bool, default=False),
                "selected_systems": Artifact(List[Path], optional=True),
                "report": Parameter(ConvReport),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        config = ip["config"]
        conv_type = config.pop("type")
        test_res = ip["test_res"]
        conv = CheckConv.get_checker(conv_type)()
        report = ConvReport()
        converged, _ = conv.check_conv(test_res, config, report)
        logging.info("Converged: %s" % converged)
        if conf_filters := ip["conf_filters_conv"]:
            logging.info("Checking filters...")
            selected_idx = conf_filters.check(test_res)
        else:
            selected_idx = list(range(len(test_res)))
        selected_systems = test_res.sub_reports(selected_idx).get_and_output_systems(
            "./systems"
        )
        report.selected_frame = len(selected_idx)
        return OPIO(
            {
                "converged": converged,
                "selected_systems": selected_systems,
                "report": report,
            }
        )
