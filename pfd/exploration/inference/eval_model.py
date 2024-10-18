from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


class EvalModel(ABC):
    def __init__(
        self,
        model: Optional[Union[Path, str]] = None,
        data: Optional[Union[Path, str]] = None,
        **kwargs
    ):
        self._data = None
        self._model = None
        if model:
            self.load_model(model)
        if data:
            self.read_data(data, **kwargs)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @abstractmethod
    def load_model(self, model: Union[Path, str], **kwargs):
        pass

    @abstractmethod
    def read_data(self, data: Union[Path, str], **kwargs):
        pass

    @abstractmethod
    def read_data_unlabeled(self, data: Union[Path, str], **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, **kwargs):
        pass

    def clear_data(self):
        self._data = None

    def clear_model(self):
        self._model = None
