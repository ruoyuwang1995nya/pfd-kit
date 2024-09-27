from typing import List

from dflow.python import OP, OPIO, Artifact, BigParameter, OPIOSign, Parameter


class StageScheduler(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "stages": Parameter(List[List[dict]]),
                "idx_stage": Parameter(int, default=0),
                #'converged':Parameter(bool,default=False)
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "tasks": Parameter(List[dict]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        return OPIO({"tasks": ip["stages"][ip["idx_stage"]]})
