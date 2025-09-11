from .check_conv import CheckConv
from dargs import Argument
from typing import List, Dict
from pathlib import Path
import logging
from pfd.exploration.inference import TestReport
from pfd.exploration.converge import ConvReport


@CheckConv.register("force_rmse_idv")
@CheckConv.register("force_rmse")
class ForceConvRMSE(CheckConv):
    def check_conv(self, reports: TestReport, config: dict, conv_report: ConvReport):
        """
        Check convergence, and selected systems for following iterations
        Args:
            test_res_ls (_type_): _description_
            conv_config (dict): _description_
            systems (List[Path]):
        Returns:
            _type_: _description_
        """
        conv_rmse = config["RMSE"]
        converged = False
        # updata convergence report
        conv_report.type = "Force_RMSE"
        conv_report.criteria = conv_rmse
        conv_report.force_rmse = reports.rmse_f
        conv_report.energy_rmse = reports.rmse_e_atom
        conv_report.selected_frame = reports.numb_frame
        # select
        logging.info(
            "#### The weighted average of force RMSE is %.6f eV/Angstrom"
                % conv_report.force_rmse
        )
        if conv_report.force_rmse < conv_rmse:
            logging.info(
                "#### Iteration converged! The converge criteria is %.6f eV/Angstrom"
                % conv_rmse
                )
            converged = True
        else:
            logging.info("#### Continue to the next iteration!")
        conv_report.converged = converged
        return converged, reports

    @classmethod
    def args(cls):
        return [
            Argument("RMSE", float, optional=True, default=0.01),
        ]

    @classmethod
    def doc(cls):
        return "Converge by RMSE of atomic forces"
