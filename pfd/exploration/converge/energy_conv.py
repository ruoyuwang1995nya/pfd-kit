from .check_conv import CheckConv
from dargs import Argument
from pfd.exploration.inference import TestReports, TestReport
from pfd.exploration.converge import ConvReport
import logging


@CheckConv.register("energy_rmse")
class EnerConvRMSE(CheckConv):
    def check_conv(self, reports: TestReport, config: dict, conv_report: ConvReport):
        """
        Check convergence, and selected systems for following iterations
        Args:
            reports (TestReports): reports of the model test
            config (dict):
        Returns:
            converged(): _description_
        """
        conv_rmse = config["RMSE"]
        converged = False
        # updata convergence report
        conv_report.type = "Energy_RMSE"
        conv_report.criteria = conv_rmse
        conv_report.force_rmse = reports.rmse_f
        conv_report.energy_rmse = reports.rmse_e_atom
        conv_report.selected_frame = reports.numb_frame
        
        
        logging.info(
                "#### The weighted average of energy RMSE per atom is %.6f eV/atom"
                % conv_report.energy_rmse
            )
        if conv_report.energy_rmse < conv_rmse:
            logging.info(
                "#### Iteration converged! The converge criteria is %.6f eV/atom"
                    % conv_rmse
                )
            converged = True
        else:
            logging.info(
                "#### Iteration not converged! Continue to the next iteration..."
                )
        conv_report.converged = converged
        return converged, reports

    @classmethod
    def args(cls):
        return [Argument("RMSE", float, optional=True, default=0.01)]

    @classmethod
    def doc(cls):
        return "Converge by RMSE of energy per atom"
