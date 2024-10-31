from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import dpdata
import numpy as np


@dataclass
class TestReport:
    name: str = "default_system"
    system: Optional[dpdata.System] = None
    atom_numb: int = 0
    numb_frame: int = 0
    mae_f: float = 0
    rmse_f: float = 0
    mae_e: float = 0
    rmse_e: float = 0
    mae_e_atom: float = 0
    rmse_e_atom: float = 0
    mae_v: float = 0
    rmse_v: float = 0
    lab_e: Optional[np.ndarray] = None
    pred_e: Optional[np.ndarray] = None
    lab_f: Optional[np.ndarray] = None
    pred_f: Optional[np.ndarray] = None
    lab_v: Optional[np.ndarray] = None
    pred_v: Optional[np.ndarray] = None

    def report(self):
        return {
            "name": self.name,
            "atom_numb": self.atom_numb,
            "numb_frame": self.numb_frame,
            "MAE_force": self.mae_f,
            "RMSE_force": self.rmse_f,
            "MAE_energy": self.mae_e,
            "RMSE_energy": self.rmse_e,
            "MAE_energy_per_at": self.mae_e_atom,
            "RMSE_energy_per_at": self.rmse_e_atom,
            "MAE_virial": self.mae_v,
            "RMSE_virial": self.rmse_v,
        }


class TestReports:
    def __init__(self, name: str = "default_reports"):
        self._reports = []
        self.name = name

    def __iter__(self):
        return iter(self._reports)

    def __getitem__(self, index):
        return self._reports[index]

    def __len__(self):
        return len(self._reports)

    def add_report(self, report: TestReport):
        self._reports.append(report)

    def get_weighted_rmse_f(self):
        if len(self._reports) > 0:
            return sum(res.numb_frame * res.rmse_f for res in self._reports) / sum(
                res.numb_frame for res in self._reports
            )

    def get_weighted_rmse_e_atom(self):
        if len(self._reports) > 0:
            return sum(res.numb_frame * res.rmse_e_atom for res in self._reports) / sum(
                res.numb_frame for res in self._reports
            )

    def get_systems(self):
        if len(self._reports) > 0:
            return [res.system for res in self._reports]
        else:
            return []

    def get_and_output_systems(self, prefix: Union[Path, str] = "."):
        if isinstance(prefix, str):
            prefix = Path(prefix)
        prefix.mkdir(exist_ok=True)
        systems = []
        for res in self._reports:
            path = prefix / res.name
            res.system.to("deepmd/npy", path)
            systems.append(path)
        return systems

    def sub_reports(self, index):
        reports = TestReports()
        if len(self._reports) > 0:
            for ii in index:
                reports.add_report(self._reports[ii])
        return reports


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
