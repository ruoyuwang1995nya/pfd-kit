from ast import Param
from typing import List, Dict
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
from pathlib import Path
from pfd.exploration.scheduler.sheduler import Scheduler
from pfd.exploration.task import ExplorationStage, BaseExplorationTaskGroup, task_group
from pfd.exploration.converge import ConvReport
from pfd.exploration import explore_styles


class StageScheduler(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "scheduler": Parameter(Scheduler),
                "systems": Artifact(List[Path]),
                "init_model": Artifact(List[Path], optional=True),
                "current_model": Artifact(List[Path], optional=True),
                "converged": Parameter(bool, value=False),
                "report": Parameter(ConvReport, value=None),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "scheduler": Parameter(Scheduler),
                "task_grp": BigParameter(BaseExplorationTaskGroup, default=None),
                "iter_numb": Parameter(int),
                "iter_id": Parameter(str),
                "next_iter_id": Parameter(str),
                "converged": Parameter(bool),
                "init_model_next": Artifact(List[Path], optional=True),
                "train_config": Parameter(Dict),
                "finetune_mode": Parameter(str, default="finetune"),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        """
        Generate exploration tasks based on model and exploration styles
        """
        systems = ip["systems"]
        scheduler = ip["scheduler"]
        converged = ip["converged"]
        init_model = ip["init_model"]
        current_model = ip["current_model"]

        if report := ip["report"]:
            scheduler.add_report(report)

        ret = {}
        ret.update({"init_model_next": init_model})
        if scheduler.rec_ft:
            ret.update({"init_model_next": current_model, "finetune_mode": "no"})
            scheduler._train_config.update(
                {
                    "init_model_with_finetune": False,
                    "init_model_policy": "yes",
                }
            )

        # check convergence
        scheduler.set_convergence(convergence_stage=converged)

        # if not converged
        if not scheduler.convergence:
            task_grp = scheduler.set_explore_tasks(systems)
            ret["task_grp"] = task_grp

        ret.update(
            {
                "scheduler": scheduler,
                "iter_numb": scheduler.iter_numb,
                "iter_id": "%03d" % scheduler.iter_numb,
                "next_iter_id": "%03d" % (scheduler.iter_numb + 1),
                "converged": scheduler.convergence,
                "train_config": scheduler.train_config,
            }
        )

        return OPIO(ret)
