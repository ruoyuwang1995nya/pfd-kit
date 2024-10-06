from abc import ABC, abstractmethod
from pathlib import Path


class CheckConv(ABC):
    @abstractmethod
    def check_conv(self):
        pass

    @classmethod
    @abstractmethod
    def doc(cls):
        return "The default method doc"

    @classmethod
    @abstractmethod
    def args(cls):
        """
        The default arguments for the method
        """
        return []
