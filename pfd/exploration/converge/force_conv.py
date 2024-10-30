from .check_conv import CheckConv
from dargs import Argument
from typing import List, Dict
from pathlib import Path
import logging
from pfd.exploration.inference import TestReports


@CheckConv.register("force_rmse_idv")
@CheckConv.register("force_rmse")
class ForceConvRMSE(CheckConv):
    def check_conv(self, reports: TestReports, config: dict):
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
        # select
        if config.get("adaptive"):
            logging.info("Adaptively add new training samples")
            selected_reports = TestReports()
            for res in reports:
                selection_thr = config.get("thr", conv_rmse)
                if res.rmse_f > selection_thr:
                    selected_reports.add_report(res)
            prec = config["adaptive"]["prec"]
            if len(reports) > 0:
                if len(selected_reports) / len(reports) > prec:
                    converged = True
        else:
            weighted_rmse = reports.get_weighted_rmse_f()
            if weighted_rmse < conv_rmse:
                converged = True
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
