from .check_conv import CheckConv
from dargs import Argument
from pfd.exploration.inference import TestReports
from pfd.exploration.converge import ConvReport
import logging


@CheckConv.register("energy_rmse")
class EnerConvRMSE(CheckConv):
    def check_conv(self, reports: TestReports, config: dict, conv_report: ConvReport):
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
        num_frame = len(reports)
        # updata convergence report
        conv_report.type = "Energy_RMSE"
        conv_report.criteria = conv_rmse
        conv_report.frame = num_frame
        conv_report.force_rmse = reports.get_weighted_rmse_f()
        conv_report.energy_rmse = reports.get_weighted_rmse_e_atom()
        # select
        if config.get("adaptive"):
            logging.info("Adaptively add new training samples")
            num_not_converged = 0
            for res in reports:
                selection_thr = config.get("thr", conv_rmse)
                if res.rmse_e_atom > selection_thr:
                    num_not_converged += 1
            prec = config["adaptive"]["prec"]
            conv_report.unconverged_frame = num_not_converged
            conv_report.criteria = prec
            if len(reports) > 0:
                if num_not_converged / num_frame > prec:
                    converged = True
        else:
            weighted_rmse = reports.get_weighted_rmse_e_atom()
            logging.info(
                "#### The weighted average of energy RMSE per atom is %.6f eV/atom"
                % weighted_rmse
            )
            if weighted_rmse < conv_rmse:
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
