from .ase import MDRunner, MDParameters
from .ase_calc import CalculatorWrapper
from .ase_replicas import ReplicaRunner, REMDParameters


# TODO: replace this with registry mechanism later.
runners = {
    "md": MDRunner,
    "replica": ReplicaRunner,
}