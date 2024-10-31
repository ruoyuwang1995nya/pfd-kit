from genericpath import isdir
from pathlib import Path
from pathlib import (
    Path,
)
from typing import List, Dict

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
from pfd.exploration.converge import CheckConv, ConfFiltersConv
from pfd.exploration.inference import TestReport, TestReports
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
        systems = ip["systems"]
        # for rep in test_res:

        # conf_filters = ip["conf_filters_conv"]
        conv = CheckConv.get_checker(conv_type)()
        converged, selected_reports = conv.check_conv(test_res, config)
        logging.info("Converged: %s" % converged)
        if conf_filters := ip["conf_filters_conv"]:
            logging.info("Checking filters...")
            selected_reports = conf_filters.check(selected_reports)
        return OPIO(
            {
                "converged": converged,
                "selected_systems": selected_reports.get_and_output_systems(
                    "./systems"
                ),
            }
        )


class NextLoop(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "converged": Parameter(bool, default=False),
                "iter_numb": Parameter(int, default=0),
                "max_iter": Parameter(int, default=1),
                "idx_stage": Parameter(int, default=0),
                "stages": Parameter(List[List[dict]]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "stage_converged": Parameter(bool, default=False),
                "converged": Parameter(bool, default=False),
                "idx_stage": Parameter(int),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        numb_stages = len(ip["stages"])
        op = {
            "converged": False,
            "stage_converged": False,
            "idx_stage": ip["idx_stage"],
        }
        if ip["iter_numb"] > ip["max_iter"]:
            op["converged"] = True
            op["stage_converged"] = True
            logging.info("Max number of iteration reached. Stop exploration...")
        elif ip["converged"] is True and ip["idx_stage"] + 1 >= numb_stages:
            op["converged"] = True
            op["stage_converged"] = True
            logging.info("All stages converged...")
        elif ip["converged"] is True and ip["idx_stage"] + 1 < numb_stages:
            op["stage_converged"] = True
            logging.info(
                "Task %s converged, continue to the next stage..." % op["idx_stage"]
            )
            op["idx_stage"] = ip["idx_stage"] + 1
        return OPIO(op)


class IterCounter(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "iter_numb": Parameter(int, default=0),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "iter_numb": Parameter(int),
                "iter_id": Parameter(str),
                "next_iter_numb": Parameter(int),
                "next_iter_id": Parameter(str),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        return OPIO(
            {
                "iter_numb": ip["iter_numb"],
                "iter_id": "%03d" % ip["iter_numb"],
                "next_iter_numb": ip["iter_numb"] + 1,
                "next_iter_id": "%03d" % (ip["iter_numb"] + 1),
            }
        )
