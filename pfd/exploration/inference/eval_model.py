from abc import ABC, abstractmethod
from pathlib import Path


class EvalModel(ABC):
    def __init__(self, model: Path | str, data: Path | str, **kwargs):
        self._data = None
        self._model = None
        self.load_model(model)
        self.read_data(data, **kwargs)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @abstractmethod
    def load_model(self, model: Path | str, **kwargs):
        pass

    @abstractmethod
    def read_data(self, data: Path | str, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass
