from abc import (
    ABC,
    abstractmethod
    )

class ConfSelect(ABC):
    @abstractmethod
    def select(self):
        pass