from typing import Dict, List
from pfd.exploration.task import ExplorationStage
from pfd.exploration.converge import ConvReport
import logging

class Scheduler:
    """
    This would manage every part of workflow status
    """

    def __init__(
        self,
        model_style: str = "dp",
        explore_style: str = "lmp",
        mass_map: List = [],
        type_map: List = [],
        explore_stages: List[ExplorationStage] = [],
        max_iter: int = 1,
        train_config: dict = {},
        finetune: bool = False,
        recursive_finetune: bool = False,
    ) -> None:

        # exploration stages
        self._explore_stages = explore_stages

        # iteration numbers
        self._idx_stage = 0

        # set iteratiion number
        self._iter_numb = 0

        # set type map
        self._type_map = type_map

        # set mass map
        self._mass_map = mass_map

        # model style
        self._model_style = model_style

        # exploration style
        self._explore_style = explore_style

        # workflow convergence
        self._converge = False

        # stage convergence
        self._converge_stage = False

        # max iterations
        self._max_iter = max_iter

        # first iteration
        self._is_first_iteration = True

        # log record
        self._log = []

        # finetune from base
        self._ft = finetune
        # recursively fintune models
        self._rec_ft = recursive_finetune

        # train config
        self._train_config = train_config

    @property
    def ft(self):
        return self._ft

    @property
    def rec_ft(self):
        return self._rec_ft

    @property
    def train_config(self):
        return self._train_config

    @property
    def model_style(self):
        return self._model_style

    @property
    def explore_style(self):
        return self._explore_style

    @property
    def expl_stages(self):
        return self._explore_stages

    @property
    def iter_numb(self):
        return self._iter_numb

    @property
    def idx_stage(self):
        return self._idx_stage

    @property
    def type_map(self):
        return self._type_map

    @property
    def mass_map(self):
        return self._mass_map

    @property
    def max_iteration(self):
        return self._max_iter

    @property
    def convergence(self):
        return self._converge

    @property
    def is_first_iteration(self):
        return self._is_first_iteration

    @property
    def log(self):
        return self._log

    @is_first_iteration.setter
    def is_first_iteration(self, value: bool):
        self._is_first_iteration = value

    
    def set_explore_tasks(self,*args,**kwargs):
        return self.expl_stages[self.idx_stage].make_task(*args, **kwargs)
        

    def set_convergence(self, convergence_stage: bool = False) -> None:
        if not self.is_first_iteration:
            self.next_iter()
        else:
            self.is_first_iteration = False
        if self.iter_numb >= self.max_iteration:
            logging.info("Max number of iteration reached. Stop exploration...")
            self._converge = True
        elif convergence_stage is True:
            if self.idx_stage + 1 >= len(self.expl_stages):
                logging.info("All stages converged...")
                self._converge = True
            else:
                logging.info(
                    "Task %s converged, continue to the next stage..." % self.idx_stage
                )
                self.next_stage()
                
        

    def next_iter(self) -> None:
        self._iter_numb += 1

    def next_stage(self) -> None:
        self._idx_stage += 1

    def add_report(self, report: ConvReport):
        report.iteration = "%03d" % self.iter_numb
        report.stage = "%03d" % self.idx_stage
        self._log.append(report)

    def get_status(self):
        from tabulate import tabulate

        if len(self.log) > 0:
            return tabulate(self.log, headers="keys", tablefmt="grid")
        else:
            return tabulate(
                self.log,
                headers=[kk for kk in ConvReport.__annotations__.keys()],
                tablefmt="grid",
            )
