from pathlib import Path
import json
from dflow.python import OP, OPIO, Artifact, BigParameter, Parameter,OPIOSign
from ase.io import read, write
from pfd.exploration.converge.check_conv import CheckConv, ConvReport
from pfd.exploration.inference import EvalModel
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ModelTestOP(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "structures": Artifact(Path), # 
                "model": Artifact(Path),
                "config": BigParameter(dict),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "test_report": Artifact(Path),
                "test_res_dir": Artifact(Path),
                "converged": Parameter(bool, default=False),
                "report": Parameter(ConvReport),  # Report on convergence
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        structures = ip["structures"]
        model_path = ip["model"]
        config = ip["config"]
        model_type = config.pop("model")
        conv_config = config.pop("converge")
        conv_type = conv_config.pop("type")
        
        ## evaluate model
        Eval = EvalModel.get_driver(model_type)
        res_dir = Path("result")
        res_dir.mkdir(exist_ok=True)
        evaluator = Eval(model_type, model=model_path, **config)
        structures = read(structures,format='extxyz',index=':')
        logging.info("##### Number of systems: %d" % len(structures))
        name = "test_model"
        if len(structures)==0:
            logging.warning("Test system is None, skipping...")
        logging.info("##### Testing..." )
        evaluator.read_data(data=structures)
        res, eval_rep = evaluator.evaluate(name, prefix=str(res_dir))
        logging.info("##### Testing ends, : writing to report...")
        
        ## check convergence
        conv = CheckConv.get_checker(conv_type)()
        conv_rep = ConvReport()
        converged, _ = conv.check_conv(res, conv_config, conv_rep)
        with open("report.json", "w") as fp:
            json.dump(eval_rep, fp, indent=4)
        return OPIO(
            {
                "test_report": Path("report.json"),
                "test_res_dir": res_dir,
                "converged": converged,
                "report": conv_rep,
            }
        )
