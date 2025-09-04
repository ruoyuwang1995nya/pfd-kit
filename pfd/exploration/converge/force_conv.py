from .check_conv import CheckConv
from dargs import Argument
from typing import List, Dict
from pathlib import Path
import logging
from pfd.exploration.inference import TestReports
from pfd.exploration.converge import ConvReport


@CheckConv.register("force_rmse_idv")
@CheckConv.register("force_rmse")
class ForceConvRMSE(CheckConv):
    def check_conv(self, reports: TestReports, config: dict, conv_report: ConvReport):
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
        num_frame = len(reports)
        # updata convergence report
        conv_report.type = "Force_RMSE"
        conv_report.criteria = conv_rmse
        conv_report.frame = num_frame
        conv_report.force_rmse = reports.get_weighted_rmse_f()
        conv_report.energy_rmse = reports.get_weighted_rmse_e_atom()
        # select
        if config.get("adaptive"):
            conv_report.type = "Force_RMSE-adaptive"
            logging.info("Adaptively add new training samples")
            num_not_converged = 0
            for res in reports:
                selection_thr = config.get("thr", conv_rmse)
                if res.rmse_f > selection_thr:
                    num_not_converged += 1
            prec = config["adaptive"]["prec"]
            conv_report.unconverged_frame = num_not_converged
            conv_report.criteria = prec
            if len(reports) > 0:
                if num_not_converged / num_frame > prec:
                    converged = True
        else:
            weighted_rmse = reports.get_weighted_rmse_f()
            logging.info(
                "#### The weighted average of force RMSE is %.6f eV/Angstrom"
                % weighted_rmse
            )
            if weighted_rmse < conv_rmse:
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
            Argument("adaptive", dict, optional=True, default=None),
        ]

    @classmethod
    def doc(cls):
        return "Converge by RMSE of atomic forces"
