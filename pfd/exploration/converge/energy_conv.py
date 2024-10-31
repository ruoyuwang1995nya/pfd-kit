from .check_conv import CheckConv
from dargs import Argument
from pfd.exploration.inference import TestReports
import logging


@CheckConv.register("energy_rmse")
class EnerConvRMSE(CheckConv):
    def check_conv(self, reports: TestReports, config: dict):
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
        return converged, reports

    @classmethod
    def args(cls):
        return [Argument("RMSE", float, optional=True, default=0.01)]

    @classmethod
    def doc(cls):
        return "Converge by RMSE of energy per atom"
