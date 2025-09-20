from abc import ABC, abstractmethod
from typing import List, Dict, Optional,Tuple
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
from pathlib import Path

from pfd.exploration.scheduler.sheduler import Scheduler
from pfd.exploration.task import  BaseExplorationTaskGroup
from pfd.exploration.converge import ConvReport


class StageScheduler(OP,ABC):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "scheduler": BigParameter(Scheduler),
                "init_model": Artifact(Path, optional=True),
                "expl_model": Artifact(Path, optional=True),  # model for exploration
                "current_model": Artifact(Path, optional=True),
                "converged": Parameter(bool, value=False),
                "report": Parameter(ConvReport, value=None),
                "optional_parameters": Parameter(Dict, default={}),
                "iter_data": Artifact(Path, optional=True),  # data collected after exploration
                #"init_data": Artifact(List[Path], optional=True),  # data collected after exploration
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "scheduler": BigParameter(Scheduler),
                "task_grp": BigParameter(BaseExplorationTaskGroup, default=None),
                "iter_numb": Parameter(int),
                "iter_id": Parameter(str),
                "next_iter_id": Parameter(str),
                "converged": Parameter(bool),
                "init_model": Artifact(Path, optional=True),
                "expl_model": Artifact(Path, optional=True),
                "current_model": Artifact(Path, optional=True),
                "iter_data": Artifact(Path, optional=True),  # data collected after exploration
                "report": Parameter(ConvReport, value=None),
                "train_config": Parameter(Dict),
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
        #systems = ip["systems"]
        scheduler = ip["scheduler"]
        converged = ip["converged"]
        init_model = ip.get("init_model")
        current_model = ip.get("current_model")
        expl_model = ip.get("expl_model")
        report = ip.get("report")
        optional_parameters = ip["optional_parameters"]

        ret = {}
        # add report if exists
        if report is not None:
            scheduler.add_report(report)

        # check convergence
        scheduler.set_convergence(convergence_stage=converged)

        # if not converged
        if not scheduler.convergence:
            task_grp = scheduler.set_explore_tasks()
            ret["task_grp"] = task_grp
            #if init_data is not None:
            #    print(type(init_data))
            #    ret["init_data"] = init_data
            #if iter_data is not None:
            #    ret["iter_data"] = iter_data
        
        ## if converged 
        #else:
        #    print("Convergence reached, no more tasks to schedule.")
        #    if init_data is None:
        #        ret["init_data"] = [iter_data]
        #    else:
        #        ret["init_data"] = init_data.append(iter_data)

        init_model, expl_model, current_model= self.schedule(
            scheduler,
            init_model=init_model,
            current_model=current_model,
            expl_model=expl_model,
            **optional_parameters
        )

        ret.update(
            {
                "scheduler": scheduler,
                "iter_numb": scheduler.iter_numb,
                "iter_id": "%03d" % scheduler.iter_numb,
                "next_iter_id": "%03d" % (scheduler.iter_numb + 1),
                "converged": scheduler.convergence,
                "train_config": scheduler.train_config,
                "init_model": init_model,
                "expl_model": expl_model,
                "current_model": current_model,
                "iter_data": ip.get("iter_data"),
                #"init_data": ip.get("init_data"),
                "report": report,
            }
        )

        return OPIO(ret)

    @abstractmethod
    def schedule(self, scheduler: Scheduler,
                 
                 *args, **kwargs):
        r"""Schedule the exploration tasks."""
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    
class StageSchedulerDist(StageScheduler):
    def schedule(
        self, 
        scheduler: Scheduler,
        #init_data: Optional[List[Path]] = None,
        #iter_data: Optional[Path] = None,
        init_model: Optional[Path] = None,
        current_model: Optional[Path] = None,
        expl_model: Optional[Path] = None,
        **kwargs) -> Tuple[Optional[Path], Optional[Path], Optional[Path],
                           #Optional[List[Path]], Optional[Path]
                           ]:
        """
        Schedule the exploration tasks in distributed mode.
        """
        return init_model, expl_model, current_model #, init_data, iter_data
    
class StageSchedulerFT(StageScheduler):
    def schedule(
        self, 
        scheduler: Scheduler,
        #init_data: Optional[List[Path]] = None,
        #iter_data: Optional[Path] = None,
        init_model: Optional[Path] = None,
        current_model: Optional[Path] = None,
        expl_model: Optional[Path] = None,
        **kwargs) -> Tuple[Optional[Path], Optional[Path], Optional[Path],
                           #Optional[List[Path]], Optional[Path]
                           ]:
        """
        Schedule the exploration tasks in distributed mode.
        """
        if current_model:
            expl_model = current_model
        if kwargs.get("iterative", False) == True:
            init_model = current_model
        return init_model, expl_model, current_model #, init_data, iter_data