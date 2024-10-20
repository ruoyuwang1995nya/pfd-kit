from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


class EvalModel(ABC):
    """The base class for inference and evaluation.

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """

    __ModelTypes = {}

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

    @staticmethod
    def register(key: str):
        """Register a model interface. Used as decorators

        Args:
            key (str): key of the model
        """

        def decorator(object):
            EvalModel.__ModelTypes[key] = object
            return object

        return decorator

    @staticmethod
    def get_driver(key: str):
        """Get a driver for ModelEval

        Args:
            key (str): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        try:
            return EvalModel.__ModelTypes[key]
        except KeyError as e:
            raise RuntimeError("unknown driver: " + key) from e

    @staticmethod
    def get_drivers() -> dict:
        """Get all drivers

        Returns:
            dict: all drivers
        """
        return EvalModel.__ModelTypes

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
