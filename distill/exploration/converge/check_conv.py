from abc import (
    ABC,
    abstractmethod
    )
from pathlib import Path

class CheckConv(ABC):
    @abstractmethod
    def check_conv(self):
        pass