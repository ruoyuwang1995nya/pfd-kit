from dataclasses import dataclass, asdict
from token import OP
from typing import Optional
import json

@dataclass
class ASEInput:
    """ASE input configuration that can be serialized."""
    #conf_str: str
    temp: float
    press: Optional[float]
    ens: str
    dt: float
    nsteps: int
    trj_freq: int
    tau_t: float
    tau_p: Optional[float]
    no_pbc: bool

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'ASEInput':
        """Deserialize from JSON string."""
        return cls(**json.loads(json_str))

    def __str__(self) -> str:
        """String representation as JSON."""
        return self.to_json()
