from pathlib import Path
from typing import (
    List,
)
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
from pfd.exploration.inference import EvalModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("infer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class InferenceOP(OP):
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
                "labeled_systems": Artifact(List[Path]),
                "root_labeled_systems": Artifact(Path),
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

        model_type = config.pop("model")
        Eval = EvalModel.get_driver(model_type)
        res_dir = Path("inference")
        res_dir.mkdir(exist_ok=True)
        labeled_systems = []
        evaluator = Eval(model=model_path)
        logging.info("##### Number of systems: %03d" % len(systems))
        for idx, sys in enumerate(systems):
            name = "sys_%03d_%s" % (idx, sys.name)
            logging.info("##### Predicting: %s" % name)
            evaluator.read_data_unlabeled(data=sys, type_map=type_map)
            _, labeled_sys = evaluator.inference(name, prefix=str(res_dir), **config)
            logging.info("##### Prediction ends")
            labeled_systems.append(labeled_sys)
        return OPIO(
            {
                "labeled_systems": labeled_systems,
                "root_labeled_systems": res_dir,
            }
        )
