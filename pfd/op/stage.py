from typing import List
from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter
from pathlib import Path
from pfd.exploration.task import ExplorationStage, BaseExplorationTaskGroup
from pfd.exploration import explore_styles


class StageScheduler(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "stages": Parameter(List[List[dict]]),
                "idx_stage": Parameter(int, default=0),
                "systems": Artifact(List[Path]),
                "type_map": Parameter(List[str]),
                "mass_map": Parameter(List[float]),
                "scheduler_config": BigParameter(dict),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "tasks": Parameter(List[dict]),
                "task_grp": BigParameter(BaseExplorationTaskGroup),
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
        expl_stage_config = ip["stages"][ip["idx_stage"]]
        type_map = ip["type_map"]
        mass_map = ip["mass_map"]
        config = ip["scheduler_config"]
        expl_stage = ExplorationStage()
        model_style = config["model_style"]
        explore_style = config["explore_style"]

        for task_grp in expl_stage_config:
            for idx in task_grp["conf_idx"]:
                expl_stage.add_task_group(
                    explore_styles[model_style][explore_style]["task"](
                        systems,
                        type_map,
                        mass_map,
                        task_grp,
                    )
                )
        return OPIO(
            {"tasks": ip["stages"][ip["idx_stage"]], "task_grp": expl_stage.make_task()}
        )
