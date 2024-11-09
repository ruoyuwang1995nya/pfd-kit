from math import log
from pathlib import Path
import json
from typing import (
    List,
)

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
from pfd.exploration.inference import EvalModel, TestReports
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ModelTestOP(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "systems": Artifact(List[Path]),
                "model": Artifact(Path),
                "type_map": Parameter(List),
                "inference_config": BigParameter(dict),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "test_report": Artifact(Path),
                "test_res": BigParameter(TestReports),
                "test_res_dir": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:

        systems = ip["systems"]
        model_path = ip["model"]
        type_map = ip["type_map"]
        config = ip["inference_config"]
        model_type = config["model"]
        Eval = EvalModel.get_driver(model_type)
        res_total = []
        report = {}
        res_dir = Path("result")
        res_dir.mkdir(exist_ok=True)
        evaluator = Eval(model=model_path)
        logging.info("##### Number of systems: %d" % len(systems))
        res_total = TestReports()
        for idx, sys in enumerate(systems):
            name = "sys_%03d_%s" % (idx, sys.name)
            logging.info("##### Testing: %s" % name)
            evaluator.read_data(data=sys, type_map=type_map)
            res, rep = evaluator.evaluate(name, prefix=str(res_dir))
            res_total.add_report(res)
            logging.info("##### Testing ends, : writing to report...")
            report[name] = rep
        with open("report.json", "w") as fp:
            json.dump(report, fp, indent=4)
        return OPIO(
            {
                "test_res": res_total,
                "test_report": Path("report.json"),
                "test_res_dir": res_dir,
            }
        )
