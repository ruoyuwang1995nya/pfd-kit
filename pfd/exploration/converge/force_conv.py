from .check_conv import CheckConv
from dargs import Argument
from typing import List, Dict
from pathlib import Path


class ForceConvIdvRMSE(CheckConv):
    def check_conv(
        self, test_res_ls: List[Dict], config: dict, systems: List[Path], **kwargs
    ):
        """
        Check convergence, and selected systems for following iterations
        Args:
            test_res_ls (_type_): _description_
            conv_config (dict): _description_
            systems (List[Path]):
        Returns:
            _type_: _description_
        """
        numb_frame = []
        rmse_e = []
        selected_systems = []
        conv_rmse = config["RMSE"]
        converged = False
        for res, sys in zip(test_res_ls, systems):
            numb_frame.append(res["numb_frame"])
            rmse_e.append(res["RMSE_force"])
            selection_thr = config.get("thr", conv_rmse)
            if res["RMSE_force"] > selection_thr:
                selected_systems.append(sys)
        if config.get("adaptive"):
            prec = config["adaptive"]["prec"]
            if len(systems) > 0:
                if len(selected_systems) / len(systems) > prec:
                    converged = True
        else:
            weighted_rmse = sum(n * rmse for n, rmse in zip(numb_frame, rmse_e)) / sum(
                numb_frame
            )
            if weighted_rmse < conv_rmse:
                converged = True
        return converged, selected_systems

    @classmethod
    def args(cls):
        return [
            Argument("RMSE", float, optional=True, default=0.01),
            Argument("adaptive", dict, optional=True, default=None),
        ]

    @classmethod
    def doc(cls):
        return "Converge by RMSE of atomic forces"


class ForceConvRMSE(CheckConv):
    def check_conv(
        self, test_res_ls: List[Dict], conv_config: dict, systems: List[Path], **kwargs
    ):
        """
        Check convergence, and selected systems for following iterations
        Args:
            test_res_ls (_type_): _description_
            conv_config (dict): _description_

        Returns:
            _type_: _description_
        """
        numb_frame = []
        rmse_e = []
        for res in test_res_ls:
            numb_frame.append(res["numb_frame"])
            rmse_e.append(res["RMSE_force"])
        conv_rmse = conv_config["RMSE"]
        weighted_rmse = sum(n * rmse for n, rmse in zip(numb_frame, rmse_e)) / sum(
            numb_frame
        )
        converged = False
        if weighted_rmse < conv_rmse:
            converged = True
        return converged, systems

    @classmethod
    def args(cls):
        return [Argument("RMSE", float, optional=True, default=0.01)]

    @classmethod
    def doc(cls):
        return "Converge by RMSE of atomic forces"
